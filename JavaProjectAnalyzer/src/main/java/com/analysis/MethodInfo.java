package com.analysis;

import java.util.List;
import java.util.stream.Collectors;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.MethodDeclaration;

public class MethodInfo {
    private String className;       // 类名（含包名）
    private String methodName;      // 方法名
    private String returnType;      // 返回类型
    private List<String> parameters;// 参数列表
    private String filePath;        // 文件路径
    private String methodBody;      // 方法体（字符串形式，可选）
    private boolean isBodyStored;   // 标记是否存储了方法体（便于内存管理）

    private MethodDeclaration methodDeclaration;

    // 构造方法（支持是否存储方法体的选项）
    // 构造方法（支持是否存储方法体的选项）
    public MethodInfo(String className, String methodName, String returnType,
                      List<Parameter> parameters, String filePath, String methodBody,
                      MethodDeclaration methodDecl, boolean storeBody) {
        this.className = className;
        this.methodName = methodName;
        this.returnType = returnType;
        this.parameters = parameters.stream()
                .map(p -> p.getType() + " " + p.getNameAsString())
                .collect(Collectors.toList());
        this.methodBody = methodBody;
        this.filePath = filePath;
        this.isBodyStored = storeBody;
        this.methodDeclaration = methodDecl;
        // 仅在需要时存储方法体（方法体可能为null，如抽象方法）
        if (storeBody && methodDecl.getBody().isPresent()) {
            this.methodBody = methodDecl.getBody().get().toString();
        } else {
            this.methodBody = null;
        }
    }

    // Getter方法
    public String getClassName() { return className; }
    public String getMethodName() { return methodName; }
    public String getMethodBody() { return methodBody; }
    public boolean isBodyStored() { return isBodyStored; }
    public String getUniqueKey() {
        return className + "." + methodName + "(" + String.join(",", parameters) + ")";
    }
    public MethodDeclaration getMethod(){
        return methodDeclaration;
    }
    @Override
    public String toString() {
        String bodyPreview = (methodBody != null)
                ? methodBody.length() > 50
                ? methodBody.substring(0, 50) + "..."  // 方法体预览（前50字符）
                : methodBody
                : "[无方法体或未存储]";

        return String.format("[%s] %s %s(%s) 位于 %s，方法体：%s",
                className, returnType, methodName,
                String.join(", ", parameters), filePath, bodyPreview);
    }
}