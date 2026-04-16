package com.analysis;
import com.entity.MethodCallTree;
import com.entity.MethodNode;
import com.github.javaparser.*;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.important.NecessaryChildAnalyzer;
import org.gradle.internal.impldep.com.google.common.base.Strings;
import org.gradle.internal.impldep.org.apache.commons.lang.ObjectUtils;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class RecursiveMethodCallParser {
    private static final Gson gson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();
    private static final List<MethodCallTree> allCallTrees = new ArrayList<>();
    private static final int MaxMethodTreeNum = 5000;
    // batch -> write the data in many times according to the MaxMethodTreeNum
    // once -> write the data in one time
    private static String mode = "once";
    private static CombinedTypeSolver typeSolver;
    private static final String jarPath = "/home/sse-wwd/.jdks/ms-21.0.9"; // 可选，设为 null 如果无 JAR 依赖
    private static final String projectName = "zaproxy";
    private static final String projectPath = "/home/sse-wwd/Codes/JavaProject/" + projectName;
//    private static final String mavenPath = "/home/sse-wwd/.m2/repository";
//    private static final String gradlePath = "/home/sse-wwd/.gradle";
    private static final List<String> sourceRoots = new ArrayList<>();

    private static final String outputJsonPath = "output";
    private static final Map<String, String> methodDeclarations = new HashMap<>();

    public static void findSubModuleSourceRoots(String projectRoot) {
        File rootDir = new File(projectRoot);
        findSourceRoots(rootDir, "src/main/java");
        findSourceRoots(rootDir, "src/test/java");
    }

    private static void findSourceRoots(File dir, String sourcePath) {
        File srcMainJava = new File(dir, sourcePath);
        if (srcMainJava.exists() && srcMainJava.isDirectory()) {
            sourceRoots.add(srcMainJava.getAbsolutePath());
        }

        File[] subDirs = dir.listFiles(File::isDirectory);
        if (subDirs != null) {
            for (File subDir : subDirs) {
                findSourceRoots(subDir, sourcePath);
            }
        }
    }
    private static void addJar(String Path){
        // 如果提供了 jarPath，添加 JarTypeSolver 用于解析 .jar 依赖

        if (Path != null && !Path.isEmpty()) {
            try (Stream<java.nio.file.Path> paths = Files.walk(Paths.get(Path))) {
                paths.filter(path -> path.toString().endsWith(".jar"))
                        .forEach(path -> {
                            try {
                                typeSolver.add(new JarTypeSolver(path));
                                // System.out.println("已注册 JAR 文件: " + path);
                            } catch (IOException e) {
                                // System.err.println("无法注册 JAR 文件: " + path + "，错误: " + e.getMessage());
                            }
                        });
            } catch (IOException e) {
                System.err.println("无法访问 JAR 目录: " + Path + "，错误: " + e.getMessage());
            }
        }

    }
    private static void configureTypeSolver() {
        // 添加 ReflectionTypeSolver 用于解析 JDK 类
        typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver());

        // 添加 JavaParserTypeSolver 用于解析项目中的自定义类型

        for(String rootPath: sourceRoots){
            typeSolver.add(new JavaParserTypeSolver(new File(rootPath)));
        }

        addJar(jarPath);
        //addJar(mavenPath);
//        addJar(gradlePath);


        // 配置 JavaParser 的 SymbolSolver
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);
        StaticJavaParser.getParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_21);
        System.out.println("dependencies finish");
    }


    // 递归解析项目目录
    private static int left = 0;
    private static int idx = 0;
    private static ArrayList<String> allFilePaths = new ArrayList<>();
    private static void parseProject(File dir, boolean register) throws InterruptedException, IOException {
        if (!dir.exists()) return;

        for (File file : Objects.requireNonNull(dir.listFiles())) {
            if (file.isDirectory()) {
                parseProject(file, register);
            } else if (file.getName().endsWith(".java")) {
                if(register){
                    registerAllMethods(file);
                    // allFilePaths.add(file.getAbsolutePath());
                    left ++;
                }else {
//                    if(file.getAbsolutePath().equals("/home/luo123456/Codes/JavaProject/smile/base/src/main/java/smile/sort/HeapSelect.java")){
//                        System.out.println("!!!");
//                        continue;
//                    }
//                    System.out.println(file.getAbsolutePath());
                    parseJavaFile(file);
                    left --;
                    if(mode.equals("batch") && allCallTrees.size() > MaxMethodTreeNum){
                        System.out.println("已经生成" + allCallTrees.size() + "个方法调用树");
                        System.out.println("剩余" + left + "个文件");
                        try (FileWriter writer = new FileWriter(outputJsonPath + idx + ".json", StandardCharsets.UTF_8)) {
                            gson.toJson(allCallTrees, writer);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }finally {
                            allCallTrees.clear();
                            idx ++;
                            Thread.sleep(1000);
                        }
                    }

                }
            }
        }
    }
    private static final ThreadLocal<StringBuilder> COMMENT_BUILDER = ThreadLocal.withInitial(StringBuilder::new);
    public static String removeComments(String methodBody) {
        if (methodBody == null || methodBody.isEmpty()) {
            return methodBody;
        }

        StringBuilder result = COMMENT_BUILDER.get();
        result.setLength(0);
        int length = methodBody.length();
        int i = 0;
        boolean inString = false;       // 是否在字符串常量中
        boolean inBlockComment = false; // 是否在多行注释（包括/** ... */）中

        while (i < length) {
            char current = methodBody.charAt(i);

            // 处理多行注释（包括/** ... */）
            if (inBlockComment) {
                // 寻找多行注释结束符 "*/"
                if (current == '*' && i + 1 < length && methodBody.charAt(i + 1) == '/') {
                    inBlockComment = false;
                    i += 2; // 跳过 "*/"
                } else {
                    i++; // 跳过注释内容（包括文档注释中的*、@等符号）
                }
                continue;
            }

            // 处理字符串常量（字符串中的/*、//、/** 均为普通字符）
            if (current == '"') {
                // 处理转义引号（\" 不视为字符串结束）
                if (i == 0 || methodBody.charAt(i - 1) != '\\') {
                    inString = !inString;
                }
                result.append(current);
                i++;
                continue;
            }

            // 非字符串区域：处理单行注释、多行注释（包括/**）
            if (!inString) {
                // 单行注释 "//"：跳过到行尾
                if (current == '/' && i + 1 < length && methodBody.charAt(i + 1) == '/') {
                    // 跳过到换行符或字符串结束
                    while (i < length && methodBody.charAt(i) != '\n') {
                        i++;
                    }
                    continue;
                }
                // 多行注释开始（包括/* 和 /**）
                else if (current == '/' && i + 1 < length && methodBody.charAt(i + 1) == '*') {
                    inBlockComment = true;
                    i += 2; // 跳过 "/*" 或 "/**" 的前两个字符（后续*会在循环中被跳过）
                    continue;
                }
            }

            // 非注释内容，添加到结果
            result.append(current);
            i++;
        }
        String cleaned = result.toString();
        COMMENT_BUILDER.remove();
        return cleaned;
    }

    // 解析单个Java文件，提取所有方法的多层调用树
    private static void parseJavaFile(File javaFile) {
        try {
            CompilationUnit cu = StaticJavaParser.parse(javaFile);
            String packageName = cu.getPackageDeclaration().map(p -> p.getNameAsString() + ".").orElse("");

            // 提取每个方法作为根节点，递归构建其调用树
            cu.findAll(MethodDeclaration.class).forEach(rootMethod -> {
                String rootClass = packageName + rootMethod.getParentNode().get().toString().split(" ")[1];
                String rootFullName = rootClass + "." + rootMethod.getNameAsString();
                String methodBody = removeComments(rootMethod.toString()).strip();
                if(rootMethod.isAbstract()){
                    return;
                }
                HashSet<String> visitedMethods = new HashSet<>();
                // 递归解析根方法的子节点（第一层子方法）
                List<MethodNode> rootChildren = parseChildMethods(rootMethod, cu, packageName, visitedMethods, 0, 10);

                // 创建根节点调用树并添加到全局列表
                MethodNode rootNode = new MethodNode(rootFullName, methodBody, null, -1, rootChildren);

                allCallTrees.add(new MethodCallTree(rootNode));
            });
            // System.out.println("finish");
        } catch (Exception e) {
            // System.err.println("解析失败：" + javaFile.getAbsolutePath() + "，错误：" + e.getMessage());
        }
    }
    private static void registerAllMethods(File javaFile){

        try {
            CompilationUnit cu = StaticJavaParser.parse(javaFile);
            // 提取每个方法作为根节点，递归构建其调用树
            cu.findAll(MethodDeclaration.class).forEach(rootMethod -> {
                methodDeclarations.putIfAbsent(rootMethod.getSignature().toString(), removeComments(rootMethod.toString()));
            });
        } catch (Exception e) {
            System.err.println("解析失败：" + javaFile.getAbsolutePath() + "，错误：" + e.getMessage());
        }
    }
    // 递归解析方法的子方法（支持多层）
    private static List<MethodNode> parseChildMethods(MethodDeclaration parentMethod, CompilationUnit cu, String packageName, HashSet<String> visitedMethods, int depth, int maxDepth) {
        if(depth == maxDepth) return null;

        List<MethodNode> childNodes = new ArrayList<>();

        // 提取当前方法的直接子方法调用
        List<MethodCallExpr> childCalls = parentMethod.findAll(MethodCallExpr.class);
        if(visitedMethods.contains(parentMethod.getSignature().toString())) return null;
        visitedMethods.add(parentMethod.getSignature().toString());

        for (MethodCallExpr childCall : childCalls) {

            // 1. 获取子方法的完整名（含包名+类名，结合resolve()逻辑）
            String childFullName = getFullMethodName(childCall, packageName);

            // 2. 判断该子方法是否为父方法的必要子方法
            boolean isNecessary = true;

            // 3. 获取调用位置（行号）
            int callLine = 0;

            // 4. 递归解析该子方法的子方法（如果能找到子方法的声明）
            List<MethodNode> grandChildNodes = new ArrayList<>();
            MethodDeclaration childMethodDecl = findMethodDeclaration(childCall, cu); // 查找子方法的声明

            if (childMethodDecl != null) {
                grandChildNodes = parseChildMethods(childMethodDecl, cu, packageName, visitedMethods, depth + 1, maxDepth); // 递归解析下一层
            }
            // 5. 创建子节点并添加到列表
            childNodes.add(new MethodNode(childFullName, methodDeclarations.getOrDefault(childCall.resolve().getSignature(), "unResolve"), isNecessary, callLine, grandChildNodes));
        }

        return childNodes;
    }

    // 辅助方法：查找子方法的声明（用于递归解析其内部调用）
    private static MethodDeclaration findMethodDeclaration(MethodCallExpr childCall, CompilationUnit cu) {
        // 简化逻辑：在当前文件中查找同名方法（实际需结合resolve()跨文件查找）
        String methodName = childCall.getNameAsString();
        return cu.findAll(MethodDeclaration.class).stream()
                .filter(m -> m.getNameAsString().equals(methodName))
                .findFirst()
                .orElse(null);
    }

    // 辅助方法：获取方法完整名（含包名+类名）
    private static String getFullMethodName(MethodCallExpr call, String packageName) {
        // 实际需通过resolve()获取准确类名（参考之前的跨类调用解析）
        String className = "";
        try{
            className = call.resolve().getClassName();
        }catch (Exception e){
            className = "unknownClass";
        }

        return packageName + className + "." + call.getNameAsString();
    }
    public static void main(String[] args) throws InterruptedException, IOException, StackOverflowError {
//        int start = 7000;
//        int end = 7680;
//        //23871
        methodDeclarations.clear();


        findSubModuleSourceRoots(projectPath);
        configureTypeSolver();

        parseProject(new File(projectPath), true);
        System.out.println("共" + left + "个文件");
//        Type listType = new TypeToken<List<String>>(){}.getType();
//        try (FileReader reader = new FileReader("allFilePaths", StandardCharsets.UTF_8)) {
//            allFilePaths = gson.fromJson(reader, listType);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        for(int i = start; i < end; i++){
//            String path = allFilePaths.get(i);
////            if(path.equals("/home/luo123456/Codes/JavaProject/elasticsearch/modules/aggregations/src/main/java/org/elasticsearch/aggregations/metric/MatrixStatsResults.java")){
////                continue;
////            }
//            // if(i == 5759) continue;
//            File file = new File(path);
//            System.out.println("正在解析：" + i);
//            parseJavaFile(file);
//        }

//        String outputPath = projectName + "_" + start + "_" + end;
//        try (FileWriter writer = new FileWriter("allFilePaths", StandardCharsets.UTF_8)) {
//            gson.toJson(allFilePaths, writer);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        Thread.sleep(2000);
//        // 解析项目并生成多层调用树
        parseProject(new File(projectPath), false);
//
        String outputPath = projectName + ".json";
//        if (mode.equals("batch")){
//            outputPath = outputJsonPath + idx + ".json";
//            idx ++;
//        }
//
//        // 写入JSON
        try (FileWriter writer = new FileWriter(outputPath, StandardCharsets.UTF_8)) {
            gson.toJson(allCallTrees, writer);
            System.out.println("生成 " + allCallTrees.size() + " 棵多层调用树，已保存至 " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
//        methodDeclarations.clear();
////
//        if(mode.equals("batch")){
//            // merge this gson
//            Type mapType = new TypeToken<List<Object>>(){}.getType();
//            List<Object> mergeData = new ArrayList<>();
//            List<File> files = new ArrayList<>();
//            files.add(new File(projectName + "_0_2000"));
//            files.add(new File(projectName + "_2000_4000"));
//            files.add(new File(projectName + "_4000_5000"));
//            files.add(new File(projectName + "_5000_5500"));
//            files.add(new File(projectName + "_5500_5700"));
//            files.add(new File(projectName + "_5700_5755"));
//            files.add(new File(projectName + "_5755_5759"));
//            files.add(new File(projectName + "_5759_5770"));
//            files.add(new File(projectName + "_5770_5800"));
//            files.add(new File(projectName + "_5800_7000"));
//            files.add(new File(projectName + "_7000_7680"));
////            files.add(new File(projectName + "_21000_23000"));
////            files.add(new File(projectName + "_23000_23871"));
//
//            for (File file : files) {
//                try (FileReader reader = new FileReader(file)) {
//                    List<Object> fileData = gson.fromJson(reader, mapType);
//                    mergeData.addAll(fileData);
//                    System.out.println("finish" + file.getName() + ".json");
//                } catch (IOException ex) {
//                    System.err.println("error " + ex.getMessage());
//                }
//            }
//            try (FileWriter writer = new FileWriter(projectName + ".json", StandardCharsets.UTF_8)) {
//                gson.toJson(mergeData, writer);
//                System.out.println("合并 " + mergeData.size() + " 棵多层调用树，已保存至 " + projectName + ".json");
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
    }

}
