package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// APIClient communicates with the llama-swap HTTP API
type APIClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewAPIClient creates a new API client, respecting LLAMA_SWAP_URL env var
func NewAPIClient(defaultURL string) *APIClient {
	url := defaultURL
	if env := os.Getenv("LLAMA_SWAP_URL"); env != "" {
		url = strings.TrimRight(env, "/")
	}
	return &APIClient{
		BaseURL: url,
		HTTPClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

// --- Data types ---

// Features describes model capabilities
type Features struct {
	Thinking bool `json:"thinking"`
	Tools    bool `json:"tools"`
	Vision   bool `json:"vision"`
}

// LlamaSwapMeta is the metadata under meta.llamaswap
type LlamaSwapMeta struct {
	Context  string   `json:"context"`
	Features Features `json:"features"`
	KvCache  string   `json:"kv_cache"`
	Size     string   `json:"size"`
	Source   string   `json:"source"`
	VRAM     string   `json:"vram_usage"`
	Warning  string   `json:"warning"`
}

// Model represents a model from /v1/models
type Model struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Meta        ModelMetaWrap `json:"meta"`
	Object      string        `json:"object"`
	OwnedBy     string        `json:"owned_by"`
}

type ModelMetaWrap struct {
	LlamaSwap LlamaSwapMeta `json:"llamaswap"`
}

// RunningModel represents a model from /running
type RunningModel struct {
	Model string `json:"model"`
	Name  string `json:"name"`
	State string `json:"state"`
	Proxy string `json:"proxy"`
	TTL   int    `json:"ttl"`
	Cmd   string `json:"cmd"`
}

// RunningResponse is the /running response
type RunningResponse struct {
	Running []RunningModel `json:"running"`
}

// ModelsResponse is the /v1/models response
type ModelsResponse struct {
	Data []Model `json:"data"`
}

// ChatMessage is a message in a chat completion request
type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content,omitempty"` // string or []contentItem for vision
}

// ChatRequest is a chat completion request
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
}

// ChatResponse is a non-streaming chat completion response
type ChatResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			Role             string `json:"role"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Timings represents llama.cpp timing statistics from streaming chunks
type Timings struct {
	PromptN           int     `json:"prompt_n"`
	PromptMS          float64 `json:"prompt_ms"`
	PredictedN        int     `json:"predicted_n"`
	PredictedMS       float64 `json:"predicted_ms"`
	PromptPerSecond   float64 `json:"prompt_per_second"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	CacheN            int     `json:"cache_n"`
}

// StreamDelta represents a streaming chunk delta
type StreamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// StreamChoice represents a choice in a streaming chunk
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        StreamDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// StreamChunk represents a streaming response chunk
type StreamChunk struct {
	ID      string         `json:"id"`
	Choices []StreamChoice `json:"choices"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Timings *Timings       `json:"timings,omitempty"`
}

// --- API methods ---

// FetchModels retrieves all configured models
func (c *APIClient) FetchModels() ([]Model, error) {
	resp, err := c.HTTPClient.Get(c.BaseURL + "/v1/models")
	if err != nil {
		return nil, fmt.Errorf("cannot reach llama-swap: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var result ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode models: %w", err)
	}

	return result.Data, nil
}

// FetchRunning retrieves currently running models
func (c *APIClient) FetchRunning() ([]RunningModel, error) {
	resp, err := c.HTTPClient.Get(c.BaseURL + "/running")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result RunningResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Running, nil
}

// UnloadModel unloads a specific model or all models
func (c *APIClient) UnloadModel(modelID string) error {
	endpoint := c.BaseURL + "/api/models/unload"
	if modelID != "" {
		endpoint += "/" + modelID
	}

	req, err := http.NewRequest("POST", endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("unload returned status %d", resp.StatusCode)
	}
	return nil
}

// FetchModelMetrics retrieves Prometheus metrics from a model's metrics endpoint
func (c *APIClient) FetchModelMetrics(port string) (map[string]float64, error) {
	url := fmt.Sprintf("http://localhost:%s/metrics", port)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	metrics := make(map[string]float64)
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		metricName := parts[0]
		if strings.HasPrefix(metricName, "llamacpp:") || strings.HasPrefix(metricName, "vllm:") {
			var val float64
			if _, err := fmt.Sscanf(parts[len(parts)-1], "%f", &val); err == nil {
				metrics[metricName] = val
			}
		}
	}
	return metrics, nil
}

// FetchSystemMetrics retrieves llama-swap system Prometheus metrics
func (c *APIClient) FetchSystemMetrics() (map[string]float64, error) {
	resp, err := c.HTTPClient.Get(c.BaseURL + "/metrics")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	metrics := make(map[string]float64)
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		if strings.HasPrefix(parts[0], "llamaswap_") {
			var val float64
			if _, err := fmt.Sscanf(parts[len(parts)-1], "%f", &val); err == nil {
				metrics[parts[0]] = val
			}
		}
	}
	return metrics, nil
}

// SendChatCompletion sends a non-streaming chat completion request
func (c *APIClient) SendChatCompletion(req ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequest("POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// StreamChatCompletion sends a streaming chat completion request
// Returns channels for chunks, errors, and timings (llama.cpp extension)
func (c *APIClient) StreamChatCompletion(req ChatRequest) (chan StreamChunk, chan error) {
	chunks := make(chan StreamChunk, 100)
	errs := make(chan error, 1)

	req.Stream = true

	body, err := json.Marshal(req)
	if err != nil {
		errs <- err
		return chunks, errs
	}

	httpReq, err := http.NewRequest("POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		errs <- err
		return chunks, errs
	}
	httpReq.Header.Set("Content-Type", "application/json")

	go func() {
		defer close(chunks)
		defer close(errs)

		client := &http.Client{Timeout: 300 * time.Second}
		resp, err := client.Do(httpReq)
		if err != nil {
			errs <- err
			return
		}
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)
		var buf bytes.Buffer

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				errs <- err
				return
			}

			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			buf.Reset()
			buf.WriteString(data)

			var chunk StreamChunk
			if err := json.Unmarshal(buf.Bytes(), &chunk); err != nil {
				continue
			}

			chunks <- chunk
		}
	}()

	return chunks, errs
}