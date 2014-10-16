/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import soot.ArrayType;
import soot.Body;
import soot.Local;
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
import soot.rtaclassload.FieldSignature;
import soot.rtaclassload.FieldSignatureUtil;
import soot.rtaclassload.MethodSignature;
import soot.rtaclassload.MethodSignatureUtil;
import soot.rtaclassload.Pair;
import soot.rtaclassload.RTAClass;
import soot.rtaclassload.RTAMethod;
import soot.rtaclassload.RTAMethodVisitor;
import soot.rtaclassload.RTAType;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.StringNumbers;
import soot.rtaclassload.StringToType;
import soot.rtaclassload.TypeToString;

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
    MethodSignature entrySignature = new MethodSignature(signature);
    for(String method : CompilerSetup.getDontDfs()){
      visited.add(method);
    }
    Set<RTAType> newInvokes = new TreeSet<RTAType>();
    newInvokes.add(entrySignature.getClassName());
    List<Pair<MethodSignature, Set<RTAType>>> entryPairs = 
        new ArrayList<Pair<MethodSignature, Set<RTAType>>>();
    entryPairs.add(new Pair<MethodSignature, Set<RTAType>>(entrySignature, newInvokes));
    Set<MethodSignature> methods = RTAClassLoader.v().callGraphFixedPoint(entryPairs);
    for(MethodSignature method : methods){
      String methodSignature = method.toString();
      searchMethod(methodSignature);
    }
  }

  private void searchMethod(String signature){
    if(visited.contains(signature)){
      return;
    }
    visited.add(signature);
    
    methodUtil.parse(signature);
    SootMethod sootMethod = methodUtil.getSootMethod();
    DfsInfo.v().addMethod(sootMethod);
    if(sootMethod.isConcrete() == false || sootMethod.isNative() == true){
      return;
    }
    if(sootMethod.hasActiveBody() == false){
      return;
    }
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
        queue.add(invokeMethod.getSignature());
      } else if(value instanceof Local){ 
        Local local = (Local) value;
        Type localType = local.getType();
        if(localType instanceof RefType){
          RefType refType = (RefType) localType;
          SootClass sootClass = refType.getSootClass();
          DfsInfo.v().addClass(sootClass);
        } else if(localType instanceof ArrayType){
          ArrayType arrayType = (ArrayType) localType;
          DfsInfo.v().addArrayType(arrayType);
        } 
      } else if(value instanceof InstanceOfExpr){
        InstanceOfExpr instanceOf = (InstanceOfExpr) value;
        DfsInfo.v().addInstanceOf(instanceOf.getCheckType());
      } else if(value instanceof NewExpr){
        NewExpr newExpr = (NewExpr) value;
        DfsInfo.v().addNewInvoke(newExpr.getType());
      }
    }
  }
}
