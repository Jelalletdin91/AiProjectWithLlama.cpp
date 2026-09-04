package com.example.llama.cppAi.configuration;

import com.openai.client.OpenAIClient;
import com.openai.client.OpenAIClientAsync;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.client.okhttp.OpenAIOkHttpClientAsync;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiAudioTranscriptionModel;
import org.springframework.ai.openai.OpenAiAudioTranscriptionOptions;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingOptions;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    public EmbeddingModel embeddingModel(){
        OpenAIClient client = OpenAIOkHttpClient.builder()
                .baseUrl("http://localhost:8081/v1")
                .apiKey("dummy")
                .build();

        return OpenAiEmbeddingModel.builder()
                .openAiClient(client)
                .options(OpenAiEmbeddingOptions.builder()
                        .model("local-embedding-model")
                        .build())
                .build();
}

    @Bean
    public VectorStore vectorStore(EmbeddingModel embeddingModel){
        return SimpleVectorStore.builder(embeddingModel).build();
    }

    @Bean
    public OpenAiAudioTranscriptionModel transcriptionModel(){
        OpenAIClient client = OpenAIOkHttpClient.builder()
                .baseUrl("http://localhost:8082/v1")
                .apiKey("dummy")
                .build();

        OpenAIClientAsync async = OpenAIOkHttpClientAsync.builder()
                .baseUrl("http://localhost:8082/v1")
                .apiKey("dummy")
                .build();

        return OpenAiAudioTranscriptionModel.builder()
                .openAiClient(client)
                .openAiClientAsync(async)
                .options(OpenAiAudioTranscriptionOptions.builder()
                        .model("whisper-1")
                        .build())
                .build();
    }

}

