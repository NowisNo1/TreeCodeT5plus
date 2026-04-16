package com.entity;

import com.google.gson.annotations.SerializedName;

import java.util.List;

// 调用树根节点容器（可选，便于区分整棵树）
public class MethodCallTree {
    @SerializedName("root")
    private MethodNode root; // 根节点（最顶层方法）

    public MethodCallTree(MethodNode root) {
        this.root = root;
    }
}

