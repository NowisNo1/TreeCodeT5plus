package com.analysis;

import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import com.analysis.ProjectEnvironmentParser.DependencyInfo;
import com.analysis.ProjectEnvironmentParser.MavenDependency;

public class DependencyDownloader {
    private final String dependencyDownloadPath;
    private final List<JarTypeSolver> jarTypeSolvers;

    public DependencyDownloader(String dependencyDownloadPath) {
        this.dependencyDownloadPath = dependencyDownloadPath;
        this.jarTypeSolvers = new ArrayList<>();
    }

    public List<JarTypeSolver> downloadAndRegisterDependencies(DependencyInfo dependencyInfo) throws Exception {
        Files.createDirectories(Paths.get(dependencyDownloadPath));

        switch (dependencyInfo.projectType) {
            case PLAIN_JAVA:
                registerLocalJars(dependencyInfo.localJars);
                break;
            case MAVEN:
                downloadAndRegisterMavenDependencies(dependencyInfo.mavenDependencies);
                break;
            case GRADLE:
                downloadAndRegisterMavenDependencies(dependencyInfo.gradleDependencies);
                break;
        }
        return jarTypeSolvers;
    }

    private void registerLocalJars(java.util.List<Path> localJars) throws IOException {
        int total = localJars.size();
        int current = 0;

        for (Path jarPath : localJars) {
            jarTypeSolvers.add(new JarTypeSolver(jarPath));
            current++;
            printFileProgressBar(current, total, "注册本地 JAR 文件");
        }
        if (total > 0) {
            System.out.println(); // 进度条完成后换行
        }
    }

    private void downloadAndRegisterMavenDependencies(java.util.List<MavenDependency> dependencies) throws IOException {
        int total = dependencies.size();
        int current = 0;

        for (ProjectEnvironmentParser.MavenDependency dependency : dependencies) {
            String jarUrl = String.format(
                    "https://repo1.maven.org/maven2/%s/%s/%s/%s-%s.jar",
                    dependency.groupId.replace(".", "/"),
                    dependency.artifactId,
                    dependency.version,
                    dependency.artifactId,
                    dependency.version
            );

            String jarFileName = String.format("%s-%s.jar", dependency.artifactId, dependency.version);
            Path downloadPath = Paths.get(dependencyDownloadPath, jarFileName);

            System.out.println("开始下载: " + jarFileName);
            downloadJar(jarUrl, downloadPath);
            jarTypeSolvers.add(new JarTypeSolver(downloadPath));
            current++;
            printFileProgressBar(current, total, "下载依赖");
        }
        if (total > 0) {
            System.out.println(); // 进度条完成后换行
        }
    }

    private void downloadJar(String jarUrl, Path downloadPath) throws IOException {
        if (Files.exists(downloadPath)) {
            System.out.println("文件已存在，跳过下载: " + downloadPath.getFileName());
            return;
        }

        HttpURLConnection connection = null;
        try {
            URL url = new URL(jarUrl);
            connection = (HttpURLConnection) url.openConnection();
            long totalSize = connection.getContentLengthLong(); // 获取文件总大小
            if (totalSize == -1) {
                // 如果无法获取文件大小，使用简单下载
                try (InputStream in = url.openStream()) {
                    Files.copy(in, downloadPath, StandardCopyOption.REPLACE_EXISTING);
                }
                System.out.println("下载完成（未知大小）: " + downloadPath.getFileName());
                return;
            }

            byte[] buffer = new byte[8192]; // 8KB 缓冲区
            long downloadedSize = 0;
            try (BufferedInputStream in = new BufferedInputStream(connection.getInputStream());
                 InputStream inputStream = in) {
                Files.createDirectories(downloadPath.getParent());
                try (var outputStream = Files.newOutputStream(downloadPath)) {
                    int bytesRead;
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                        downloadedSize += bytesRead;
                        printFileProgressBar(downloadedSize, totalSize, downloadPath.getFileName().toString());
                    }
                }
            }
            System.out.println("\n下载完成: " + downloadPath.getFileName());
        } catch (IOException e) {
            System.err.println("下载 JAR 文件失败 " + jarUrl + ": " + e.getMessage());
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    private void printFileProgressBar(long currentBytes, long totalBytes, String fileName) {
        int barLength = 50; // 进度条长度
        double progress = (double) currentBytes / totalBytes;
        int filled = (int) (progress * barLength);
        StringBuilder bar = new StringBuilder("[");

        // 构建进度条
        for (int i = 0; i < barLength; i++) {
            if (i < filled) {
                bar.append("=");
            } else if (i == filled && currentBytes < totalBytes) {
                bar.append(">");
            } else {
                bar.append(" ");
            }
        }
        bar.append("] ");

        // 计算百分比和文件大小
        int percentage = (int) (progress * 100);
        DecimalFormat df = new DecimalFormat("#.1");
        String currentMB = df.format(currentBytes / 1024.0 / 1024.0);
        String totalMB = df.format(totalBytes / 1024.0 / 1024.0);
        bar.append(String.format("%d%% (%s MB/%s MB) %s", percentage, currentMB, totalMB, fileName));

        // 打印进度条，覆盖前一行
        System.out.print("\r" + bar.toString());
    }
}