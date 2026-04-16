package com.entity;

import com.google.gson.annotations.SerializedName;

import java.util.List;

// 递归节点：既能表示父方法，也能表示任意层级的子方法
public class MethodNode {
    @SerializedName("method_full_name")
    private String methodFullName; // 方法完整名（含包名+类名+方法名，如"com.example.UserService.getUser"）

    @SerializedName("method_body")
    private String methodBody;
    @SerializedName("is_necessary")
    private Boolean isNecessary; // 仅对非根节点有效：是否为父方法的必要子方法（null表示根节点）

    @SerializedName("call_position")
    private int callPosition; // 在父方法中的调用行号（根节点为-1）

    @SerializedName("children")
    private List<MethodNode> children; // 子方法节点（递归结构，支持任意深度）


    // 构造方法
    public MethodNode(String methodFullName, String methodBody, Boolean isNecessary, int callPosition, List<MethodNode> children) {
        this.methodFullName = methodFullName;
        this.methodBody = methodBody;
        this.isNecessary = isNecessary;
        this.callPosition = callPosition;
        this.children = children;
    }

    // Getter/Setter省略
}
