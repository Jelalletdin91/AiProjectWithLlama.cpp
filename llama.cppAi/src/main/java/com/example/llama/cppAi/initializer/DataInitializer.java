package com.example.llama.cppAi.initializer;

import jakarta.annotation.PostConstruct;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

@Component
public class DataInitializer {

    @Autowired
    private VectorStore vectorStore;

    @PostConstruct
    public void initData() throws IOException {
        ClassPathResource resource = new ClassPathResource("Shakikh_Al_Bukhari.txt");
        String content = new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

        List<Document> khadises = Arrays.stream(content.split("\\r?\\n\\s*\\r?\\n"))
                .map(String::trim)
                .filter(block -> !block.isEmpty())
                .map(Document::new)
                .toList();

        vectorStore.add(khadises);
    }

}
