package com.analysis;

import org.apache.maven.model.Model;
import org.apache.maven.model.io.xpp3.MavenXpp3Reader;
import org.gradle.tooling.GradleConnector;
import org.gradle.tooling.ProjectConnection;
import org.gradle.tooling.model.eclipse.EclipseClasspathContainer;
import org.gradle.tooling.model.eclipse.EclipseClasspathEntry;
import org.gradle.tooling.model.eclipse.EclipseProject;
import org.gradle.tooling.model.gradle.GradleBuild;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ProjectEnvironmentParser {
    private final String projectRoot;
    private static final String ALIYUN_GRADLE_MIRROR_BASE = "https://mirrors.aliyun.com/gradle/distributions/";
    public ProjectEnvironmentParser(String projectRoot) {
        this.projectRoot = projectRoot;
    }

    public enum ProjectType {
        PLAIN_JAVA, MAVEN, GRADLE
    }

    public static class DependencyInfo {
        public final ProjectType projectType;
        public final List<Path> localJars; // For PLAIN_JAVA
        public final List<MavenDependency> mavenDependencies; // For MAVEN
        public final List<MavenDependency> gradleDependencies; // For GRADLE

        public DependencyInfo(ProjectType projectType) {
            this.projectType = projectType;
            this.localJars = new ArrayList<>();
            this.mavenDependencies = new ArrayList<>();
            this.gradleDependencies = new ArrayList<>();
        }
    }

    public static class MavenDependency {
        public final String groupId;
        public final String artifactId;
        public final String version;

        public MavenDependency(String groupId, String artifactId, String version) {
            this.groupId = groupId;
            this.artifactId = artifactId;
            this.version = version;
        }
    }

    public DependencyInfo parseEnvironment() throws Exception {
        ProjectType projectType = detectProjectType();
        DependencyInfo dependencyInfo = new DependencyInfo(projectType);

        switch (projectType) {
            case PLAIN_JAVA:
                parsePlainJavaProject(dependencyInfo);
                break;
            case MAVEN:
                parseMavenProject(dependencyInfo);
                break;
            case GRADLE:
                replaceGradleDistributionUrl(projectRoot);
                parseGradleProject(dependencyInfo);
                break;
        }
        return dependencyInfo;
    }

    private ProjectType detectProjectType() {
        File pomFile = new File(projectRoot, "pom.xml");
        File buildGradle = new File(projectRoot, "build.gradle");

        if (pomFile.exists()) {
            return ProjectType.MAVEN;
        } else if (buildGradle.exists()) {
            return ProjectType.GRADLE;
        } else {
            return ProjectType.PLAIN_JAVA;
        }
    }

    private void parsePlainJavaProject(DependencyInfo dependencyInfo) throws IOException {
        Files.walkFileTree(Paths.get(projectRoot), new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                if (file.toString().endsWith(".jar")) {
                    dependencyInfo.localJars.add(file);
                }
                return FileVisitResult.CONTINUE;
            }
        });
    }

    private void parseMavenProject(DependencyInfo dependencyInfo) throws Exception {
        File pomFile = new File(projectRoot, "pom.xml");
        MavenXpp3Reader reader = new MavenXpp3Reader();
        Model model = reader.read(new FileReader(pomFile));

        for (org.apache.maven.model.Dependency dependency : model.getDependencies()) {
            String groupId = dependency.getGroupId();
            String artifactId = dependency.getArtifactId();
            String version = dependency.getVersion();

            if (version == null || version.isEmpty()) {
                System.out.println("Skipping dependency " + groupId + ":" + artifactId + " due to missing version");
                continue;
            }

            dependencyInfo.mavenDependencies.add(new MavenDependency(groupId, artifactId, version));
        }
    }

    private void parseGradleProject(DependencyInfo dependencyInfo) throws Exception {
        try {
            GradleConnector connector = GradleConnector.newConnector();
            connector.forProjectDirectory(new File(projectRoot));
            try (ProjectConnection connection = connector.connect()) {
                // 获取项目基本信息
                GradleBuild buildModel = connection.getModel(GradleBuild.class);
                System.out.println("分析项目: " + buildModel.getProjects().iterator().next().getName());

                // 获取依赖信息
                EclipseProject eclipseProject = connection.getModel(EclipseProject.class);
                System.out.println("?");
                for (EclipseClasspathContainer container : eclipseProject.getClasspathContainers()) {
                    System.out.println(container);
                }


            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    /**
     * 从项目根目录查找 gradle-wrapper.properties 文件
     *
     * @param projectRootPath 项目根目录路径
     * @return 文件的绝对路径，未找到时返回 null
     * @throws IOException 如果搜索过程中发生错误
     */
    public static Path findGradleWrapperProperties(String projectRootPath) throws IOException {
        Path root = Paths.get(projectRootPath);
        try (Stream<Path> stream = Files.walk(root, FileVisitOption.FOLLOW_LINKS)) {
            List<Path> result = stream
                    .filter(p -> p.getFileName().toString().equals("gradle-wrapper.properties"))
                    .collect(Collectors.toList());

            Path of = Path.of("");
            if (result.isEmpty()) {
                return of; // 未找到文件
            }

            // 返回最顶层的匹配文件（最短路径）
            return result.stream()
                    .min(Comparator.comparingInt(Path::getNameCount))
                    .orElse(of);
        }
    }
    /**
     * 替换 Gradle wrapper 配置文件中的 distributionUrl 为阿里云镜像地址，
     * 并将结果保存为 .txt 文件，不修改原始文件
     *
     * @param propertiesFilePath gradle-wrapper.properties 文件的路径
     * @throws IOException              如果读取或写入文件时发生错误
     * @throws IllegalArgumentException 如果文件中未找到有效的 distributionUrl
     */
    public void replaceGradleDistributionUrl(String propertiesFilePath) throws IOException {
        Path originalPath = findGradleWrapperProperties(propertiesFilePath);

        if (!Files.exists(originalPath)) {
            throw new FileNotFoundException("未找到文件: " + propertiesFilePath);
        }

        // 生成 TXT 备份文件路径（在原文件同目录下，后缀改为 .txt）
        Path txtPath = originalPath.resolveSibling(
                originalPath.getFileName().toString().replace(".properties", ".txt")
        );

        // 读取原始文件内容
        List<String> lines = Files.readAllLines(originalPath, StandardCharsets.UTF_8);

        // 写入 TXT 备份文件
        Files.write(txtPath, lines, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);

        boolean replaced = false;

        // 处理每一行，替换 distributionUrl
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            // 匹配 distributionUrl 行（忽略前后空格）
            if (line.trim().startsWith("distributionUrl=")) {
                // 正则提取版本号和文件名
                Pattern pattern = Pattern.compile("distributionUrl=.*/gradle-(.+?)-bin\\.zip");
                Matcher matcher = pattern.matcher(line);

                if (matcher.matches()) {
                    String version = matcher.group(1);
                    // 构建阿里云镜像 URL（修正后的正确格式）
                    String mirrorUrl = "distributionUrl=" + ALIYUN_GRADLE_MIRROR_BASE + "v" + version + ".0/" + "gradle-" + version + "-bin.zip";
                    lines.set(i, mirrorUrl);
                    replaced = true;
                    break;
                }
            }
        }

        if (!replaced) {
            throw new IllegalArgumentException("文件中未找到有效的 distributionUrl 配置");
        }


        Files.write(originalPath, lines, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);

    }

}