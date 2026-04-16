package com.important;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AssignExpr; // 3.27.0正确类名：AssignExpr
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.WhileStmt;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

public class NecessaryChildAnalyzer {
    // 主方法：判断子方法是否必要
    private final Set<MethodDeclaration> visitedMethods = new HashSet<>();
    public boolean isNecessaryChild(MethodDeclaration parentMethod, MethodCallExpr childCall) {
        boolean affectsReturnValue = checkReturnValueDependency(parentMethod, childCall);
        boolean affectsKeyVariable = checkVariableDependency(parentMethod, childCall);
        boolean affectsControlFlow = checkControlFlowDependency(childCall);

        return affectsReturnValue || affectsKeyVariable || affectsControlFlow;
    }
    public void clear(){
        visitedMethods.clear();
    }
    // 条件1：检查返回值依赖
    private boolean checkReturnValueDependency(MethodDeclaration parent, MethodCallExpr child) {
        // 调用findAll(ReturnStmt.class)时，添加注解消除警告
        List<ReturnStmt> returnStmts = parent.findAll(ReturnStmt.class);
        String childName = child.getNameAsString();

        for (ReturnStmt returnStmt : returnStmts) {
            Optional<Expression> returnExpr = returnStmt.getExpression();
            if (returnExpr.isPresent()) {
                // 判断子方法名是否出现在返回表达式中（更健壮）
                if (returnExpr.get().toString().contains(childName)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 条件2：检查变量依赖（使用3.27.0的AssignExpr）
    @SuppressWarnings("unchecked")
    private boolean checkVariableDependency(MethodDeclaration parent, MethodCallExpr child) {

        Optional<AssignExpr> assignment = child.findAncestor(AssignExpr.class);
        if (assignment.isEmpty()) {
            return false; // 子方法未参与赋值，不影响变量
        }

        // 获取被赋值的变量名（如"x"）
        Expression targetExpr = assignment.get().getTarget();
        String varName = targetExpr.toString().trim();
        if (varName.isEmpty()) {
            return false;
        }

        // 检查变量是否被其他必要子方法使用
        List<MethodCallExpr> otherCalls = parent.findAll(MethodCallExpr.class);
        for (MethodCallExpr otherCall : otherCalls) {
            if (otherCall.equals(child)) continue;
            // 递归判断其他子方法是否必要，且参数包含该变量
            if(visitedMethods.contains(parent)) continue;
            visitedMethods.add(parent);
            if (isNecessaryChild(parent, otherCall) && otherCall.toString().contains(varName)) {
                return true;
            }
        }
        return false;
    }

    // 条件3：检查控制流依赖
    @SuppressWarnings("unchecked")
    private boolean checkControlFlowDependency(MethodCallExpr child) {
        // 检查子方法是否在if/while/for条件中
        boolean inIf = child.findAncestor(IfStmt.class).isPresent();
        boolean inWhile = child.findAncestor(WhileStmt.class).isPresent();
        boolean inFor = child.findAncestor(ForStmt.class).isPresent();

        return inIf || inWhile || inFor;
    }

}