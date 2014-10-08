/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import soot.ArrayType;
import soot.Body;
import soot.RefType;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Type;
import soot.Value;
import soot.ValueBox;
import soot.jimple.FieldRef;
import soot.jimple.InstanceOfExpr;
import soot.jimple.InvokeExpr;
import soot.jimple.NewExpr;
import soot.rbclassload.FieldSignature;
import soot.rbclassload.FieldSignatureUtil;
import soot.rbclassload.MethodSignature;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RTAClass;
import soot.rbclassload.RTAMethod;
import soot.rbclassload.RTAMethodVisitor;
import soot.rbclassload.RTAType;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;
import soot.rbclassload.StringToType;

public class RootbeerDfs {
  
  private Set<String> visited;
  private LinkedList<String> queue;
  private MethodSignatureUtil methodUtil;
  
  public RootbeerDfs(){
    visited = new HashSet<String>();
    queue = new LinkedList<String>();
    methodUtil = new MethodSignatureUtil();
  }
  
  public void run(String signature) {
    
    queue.add(signature);
    CompilerSetup setup = new CompilerSetup();
    for(String method : setup.getDontDfs()){
      visited.add(method);
    }
    
    while(queue.isEmpty() == false){
      String curr = queue.removeFirst();
      searchMethod(curr);
    }
  }

  private void searchMethod(String signature){
    if(visited.contains(signature)){
      return;
    }
    visited.add(signature);
    
    methodUtil.parse(signature);
    SootMethod sootMethod = methodUtil.getSootMethod();
    if(sootMethod.isConcrete() == false){
      return;
    }
    System.out.println("  searchMethod: "+sootMethod.getSignature());
    Body body = sootMethod.getActiveBody();
    List<ValueBox> values = body.getUseAndDefBoxes();
    for(ValueBox box : values){
      Value value = box.getValue();
      if(value instanceof FieldRef){
        FieldRef fieldRef = (FieldRef) value;
        SootField sootField = fieldRef.getField();
        DfsInfo.v().addField(sootField);
      } else if(value instanceof InvokeExpr){
        InvokeExpr invokeExpr = (InvokeExpr) value;
        SootMethod invokeMethod = invokeExpr.getMethod();
        DfsInfo.v().addMethod(invokeMethod);
      } else if(value instanceof RefType){
        RefType refType = (RefType) value;
        SootClass sootClass = refType.getSootClass();
        DfsInfo.v().addClass(sootClass);
      } else if(value instanceof InstanceOfExpr){
        InstanceOfExpr instanceOf = (InstanceOfExpr) value;
        DfsInfo.v().addInstanceOf(instanceOf.getType());
      } else if(value instanceof ArrayType){
        ArrayType arrayType = (ArrayType) value;
        DfsInfo.v().addArrayType(arrayType);
      } else if(value instanceof NewExpr){
        NewExpr newExpr = (NewExpr) value;
        DfsInfo.v().addNewInvoke(newExpr.getType());
      }
    }
  }
}
