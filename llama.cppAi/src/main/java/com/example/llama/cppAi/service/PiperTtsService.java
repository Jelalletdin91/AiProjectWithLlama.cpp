package com.example.llama.cppAi.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * Синтез речи через Piper.
 *
 * У Piper нет режима HTTP-сервера, это консольная программа.
 * Поэтому запускаем её как процесс: текст подаём в stdin,
 * готовый WAV забираем из временного файла.
 */
@Service
public class PiperTtsService {

    private static final Logger log = LoggerFactory.getLogger(PiperTtsService.class);

    @Value("${piper.exe}")
    private String piperExe;

    @Value("${piper.model}")
    private String piperModel;

    /** Максимальное время на синтез одной фразы. */
    private static final int TIMEOUT_SECONDS = 60;

    public byte[] synthesize(String text) throws IOException, InterruptedException {

        if (text == null || text.isBlank()) {
            return new byte[0];
        }

        // Piper пишет результат в файл, а не в stdout —
        // так надёжнее, чем разбирать поток вперемешку с логами
        Path wav = Files.createTempFile("piper-", ".wav");

        try {
            ProcessBuilder pb = new ProcessBuilder(
                    piperExe,
                    "-m", piperModel,
                    "-f", wav.toString(),
                    "--length_scale", "0.8",
                    "--noise_scale", "0.8"
            );
            // ошибки Piper пойдут в тот же поток, что и вывод —
            // так их видно в логах при отладке
            pb.directory(new java.io.File(piperExe).getParentFile());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            // Текст передаём в stdin. Кодировку задаём явно —
            // иначе турецкие ş, ğ, ı превратятся в мусор.
            try (var writer = new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8)) {
                writer.write(text);
            }

            // читаем вывод процесса, чтобы буфер не переполнился и не заблокировал его
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new IOException("Piper не ответил за " + TIMEOUT_SECONDS + " секунд");
            }

            if (process.exitValue() != 0) {
                throw new IOException("Piper завершился с ошибкой: " + output);
            }

            byte[] audio = Files.readAllBytes(wav);
            log.debug("Piper: {} символов -> {} байт аудио", text.length(), audio.length);
            return audio;

        } finally {
            Files.deleteIfExists(wav);
        }
    }
}