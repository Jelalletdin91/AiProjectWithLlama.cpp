package com.example.llama.cppAi.controller;

import com.example.llama.cppAi.service.PiperTtsService;
import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiAudioSpeechModel;
import org.springframework.ai.openai.OpenAiAudioTranscriptionModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.stream.Collectors;

@RestController
public class LlamaController {

    private ChatClient chatClient;

    private PiperTtsService piperTts;

    @Autowired
    private EmbeddingModel embeddingModel;

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private OpenAiAudioTranscriptionModel aiModel;



     ChatMemory chatMemory = MessageWindowChatMemory.builder().build();

    public LlamaController(ChatClient.Builder builder, OpenAiAudioTranscriptionModel aiModel, PiperTtsService piperTts){
        this.chatClient=builder.defaultSystem("""
                Sen çocuklara yardımcı olan bir robotsun. Adın Jarvis.
                
                BİLGİ KURALLARI
                - Sadece sana verilen metne dayanarak cevap ver.
                - Verilen metin soruyla ilgisizse ya da yetersizse şöyle de:
                  "Bunu bilmiyorum, birlikte öğrenebiliriz." Tahmin yürütme.
                - Hadis ya da ayet uydurma. Metinde ne yazıyorsa onu sade dille anlat.
                - Soru dinle ilgili değilse normal ve kısa cevap ver, zorla hadise bağlama.
                
                NASIL KONUŞMALISIN
                - Karşındaki yedi ile on iki yaş arası bir çocuk. Sıcak, sade ve kısa konuş.
                - En fazla üç cümle.
                - Zor bir kelime kullanman gerekirse hemen ardından basitçe açıkla.
                
                SESLİ OKUMA KURALLARI
                Cevabın bir ses motoru tarafından okunacak. Bu yüzden:
                - Yıldız, tire, madde imi, başlık, emoji kullanma. Düz cümle yaz.
                - Nokta içeren kısaltma kullanma. "Hz." yerine "Hazreti", "vb." yerine
                  "gibi şeyler", "örn." yerine "mesela" yaz.
                - Arap harfleriyle yazma. Gerekirse Türkçe okunuşunu ver.
                - Kaynağı sadece sorulduğunda söyle.
                
                Türkçe cevap ver.
                    """).defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build()).build();
        this.aiModel=aiModel;
        this.piperTts=piperTts;
    }

    @GetMapping("api/chat/{text}")
    public String getAns(@PathVariable String text){
        return chatClient.prompt(text)
                .advisors(a->a.param(ChatMemory.CONVERSATION_ID, "default_chat"))
                .call()
                .content();
    }

    @PostMapping("api/embedding")
    public float[] embedding(@RequestParam String text){
        return embeddingModel.embed(text);
    }

    @PostMapping("/api/similarity")
    public double similarity(@RequestParam String text1, @RequestParam String text2){
        float[] embedding1 = embeddingModel.embed(text1);
        float[] embedding2 = embeddingModel.embed(text2);

        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;

        for (int i = 0; i<embedding1.length; i++){
            dotProduct+=embedding1[i] * embedding2[i];
            norm1+= Math.pow(embedding1[i], 2);
            norm2+= Math.pow(embedding2[i], 2);
        }

        return dotProduct*100 / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    public record SearchHit(String text, Double score) {}

    @GetMapping("api/search/{text}")
    public List<SearchHit> search(@PathVariable String text){
        List<Document> found = vectorStore.similaritySearch(SearchRequest.builder().query(text).topK(5).build());

        return found.stream().map(d -> new SearchHit(d.getText(), d.getScore())).toList();
    }

    @GetMapping(value = "api/ask/{text}", produces = MediaType.TEXT_PLAIN_VALUE + ";charset=UTF-8")
    public Flux<String> ask(@PathVariable String text){
        List<Document> found = vectorStore.similaritySearch(SearchRequest.builder().query(text).topK(2).similarityThreshold(0.6).build());

        if (found.isEmpty()){
            return chatClient.prompt(text).advisors(a->a.param(ChatMemory.CONVERSATION_ID, "main_chat"))
                    .stream()
                    .content();
        }

        String context = found.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n\n"));

        return chatClient.prompt()
                .system("""
        Sen bir hadis asistanısın.
        HER ZAMAN Türkçe cevap ver. Cevapların en fazla 2-5 cümle olsun.

        Aşağıdaki hadislere dayanarak cevap ver.
        Soru hadislerle ilgili değilse, kendi bilginle cevap ver.

        Hadisler:
        %s
        """.formatted(context))
                .user(text)                     // ← в память попадёт только это
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "main_chat"))
                .stream()
                .content();
    }

    @PostMapping("api/stt")
    public String getText(@RequestParam MultipartFile file){
        return aiModel.call(new AudioTranscriptionPrompt(file.getResource()))
                .getResult()
                .getOutput();
    }

    @PostMapping(value = "/api/tts", produces = "audio/wav")
    public ResponseEntity<byte[]> textToSpeech(@RequestParam String text) {
        try {
            return ResponseEntity.ok(piperTts.synthesize(text));
        } catch (Exception e) {
            // не роняем запрос — интерфейс просто останется без озвучки
            return ResponseEntity.internalServerError().build();
        }
    }


}
