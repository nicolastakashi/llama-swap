package genaimetrics

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"github.com/mostlygeek/llama-swap/internal/hint"
	"github.com/mostlygeek/llama-swap/internal/store"
)

func envOf(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestGenAIMetrics_New(t *testing.T) {
	m, err := New(context.Background(), envOf(map[string]string{"OTEL_SDK_DISABLED": "true"}), "v1", nil)
	require.NoError(t, err)
	assert.Nil(t, m)

	// The SDK reads its settings from the real environment. A malformed
	// resource attribute (an entry with no value) must not turn metrics off,
	// and OTLP is added only when an endpoint is set.
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
	t.Setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=homelab,oops")
	for name, env := range map[string]map[string]string{
		"prometheus only": {},
		"with otlp": {
			"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",
			"OTEL_METRICS_EXPORTER":       "console, OTLP",
		},
	} {
		m, err := New(context.Background(), envOf(env), "v1", nil)
		require.NoError(t, err, name)
		require.NotNil(t, m, name)
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // don't wait on the unreachable endpoint
		_ = m.Shutdown(ctx)
	}
}

// TestGenAIMetrics_Prometheus checks the text llama-swap appends to
// /metrics: Prometheus names translated from the semconv ones, labels and
// the advised buckets.
func TestGenAIMetrics_Prometheus(t *testing.T) {
	m, err := New(context.Background(), envOf(nil), "v1", nil)
	require.NoError(t, err)
	defer m.Shutdown(context.Background())

	m.Record(store.ActivityLogEntry{
		Model: "qwen", Task: "chat", ReqPath: "/v1/chat/completions", RespStatusCode: 200, DurationMs: 1500,
		Tokens: store.TokenMetrics{InputTokens: 100, OutputTokens: 50, TokensPerSecond: 20},
	}, hint.LlamaServer)

	var buf bytes.Buffer
	require.NoError(t, m.WritePrometheus(&buf))
	out := buf.String()
	labels := `gen_ai_operation_name="chat",gen_ai_provider_name="llama.cpp",gen_ai_request_model="qwen"`
	for _, want := range []string{
		"# TYPE gen_ai_server_request_duration_seconds histogram",
		"gen_ai_server_request_duration_seconds_bucket{" + labels + `,le="2.56"} 1`,
		"gen_ai_server_request_duration_seconds_count{" + labels + "} 1",
		"gen_ai_client_token_usage_bucket{" + labels + `,gen_ai_token_type="input",le="256"} 1`,
		"gen_ai_server_time_per_output_token_seconds_sum{" + labels + "} 0.05",
		`target_info{service_name="llama-swap"`,
	} {
		assert.Contains(t, out, want)
	}
}

// TestGenAIMetrics_Record checks names, attributes and bucket placement of
// what a few requests produce.
func TestGenAIMetrics_Record(t *testing.T) {
	reader := sdkmetric.NewManualReader()
	m, err := newMetrics(sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader)))
	require.NoError(t, err)

	chat := store.ActivityLogEntry{
		Model: "qwen", Task: "chat", ReqPath: "/v1/chat/completions", RespStatusCode: 200, DurationMs: 1500,
		Tokens: store.TokenMetrics{InputTokens: 100, OutputTokens: 50, TokensPerSecond: 20},
	}
	m.Record(chat, hint.LlamaServer)
	m.Record(chat, hint.LlamaServer)
	// A failed request: zeroed tokens are not usage, and no decode rate.
	m.Record(store.ActivityLogEntry{Model: "qwen", Task: "chat", ReqPath: "/v1/completions", RespStatusCode: 503, DurationMs: 10}, hint.Unknown)
	// A client hang-up is not a server error.
	m.Record(store.ActivityLogEntry{Model: "aborted", Task: "chat", ReqPath: "/v1/chat/completions", RespStatusCode: 499, DurationMs: 10}, hint.LlamaServer)

	var rm metricdata.ResourceMetrics
	require.NoError(t, reader.Collect(context.Background(), &rm))
	require.Len(t, rm.ScopeMetrics, 1)

	byName := map[string]metricdata.Metrics{}
	for _, metric := range rm.ScopeMetrics[0].Metrics {
		byName[metric.Name] = metric
	}
	require.Len(t, byName, 3)
	assert.Equal(t, "s", byName["gen_ai.server.request.duration"].Unit)
	assert.Equal(t, "{token}", byName["gen_ai.client.token.usage"].Unit)

	ok := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.provider.name", "llama.cpp"),
		attribute.String("gen_ai.request.model", "qwen"),
	}
	with := func(kv ...attribute.KeyValue) attribute.Set { return attribute.NewSet(append(kv, ok...)...) }

	// 1.5 s falls in (1.28, 2.56], bucket 8.
	duration := points[float64](t, byName["gen_ai.server.request.duration"])
	require.Len(t, duration, 3)
	d := duration[with()]
	assert.Equal(t, uint64(2), d.Count)
	assert.Equal(t, 3.0, d.Sum)
	assert.Equal(t, uint64(2), d.BucketCounts[8])
	assert.Equal(t, durationBounds, d.Bounds)

	failed := duration[attribute.NewSet(
		attribute.String("gen_ai.operation.name", "text_completion"),
		attribute.String("gen_ai.provider.name", "unknown"),
		attribute.String("gen_ai.request.model", "qwen"),
		attribute.String("error.type", "503"),
	)]
	assert.Equal(t, uint64(1), failed.Count)

	aborted := duration[attribute.NewSet(ok[0], ok[1], attribute.String("gen_ai.request.model", "aborted"))]
	assert.Equal(t, uint64(1), aborted.Count)

	tokens := points[int64](t, byName["gen_ai.client.token.usage"])
	require.Len(t, tokens, 2)
	assert.Equal(t, int64(200), tokens[with(attribute.String("gen_ai.token.type", "input"))].Sum)
	assert.Equal(t, int64(100), tokens[with(attribute.String("gen_ai.token.type", "output"))].Sum)

	// 1/20 tok/s = 0.05 s sits exactly on a bound, so (0.025, 0.05], bucket 2.
	tpot := points[float64](t, byName["gen_ai.server.time_per_output_token"])
	require.Len(t, tpot, 1)
	assert.Equal(t, uint64(2), tpot[with()].BucketCounts[2])
}

func points[N int64 | float64](t *testing.T, m metricdata.Metrics) map[attribute.Set]metricdata.HistogramDataPoint[N] {
	t.Helper()
	h, ok := m.Data.(metricdata.Histogram[N])
	require.True(t, ok, "%s is %T", m.Name, m.Data)
	assert.Equal(t, metricdata.CumulativeTemporality, h.Temporality)
	out := map[attribute.Set]metricdata.HistogramDataPoint[N]{}
	for _, p := range h.DataPoints {
		out[p.Attributes] = p
	}
	return out
}

func TestGenAIMetrics_ExporterProtocol(t *testing.T) {
	for protocol, want := range map[string]string{
		"":              "*otlpmetrichttp.Exporter",
		"http/protobuf": "*otlpmetrichttp.Exporter",
		"http/json":     "*otlpmetrichttp.Exporter", // unsupported: falls back with a warning
		"grpc":          "*otlpmetricgrpc.Exporter",
	} {
		exp, err := newExporter(context.Background(), envOf(map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": protocol}), nil)
		require.NoError(t, err, protocol)
		assert.Equal(t, want, fmt.Sprintf("%T", exp), protocol)
	}

	// The signal-specific variable wins.
	exp, err := newExporter(context.Background(), envOf(map[string]string{
		"OTEL_EXPORTER_OTLP_PROTOCOL":         "http/protobuf",
		"OTEL_EXPORTER_OTLP_METRICS_PROTOCOL": "grpc",
	}), nil)
	require.NoError(t, err)
	assert.Equal(t, "*otlpmetricgrpc.Exporter", fmt.Sprintf("%T", exp))
}
