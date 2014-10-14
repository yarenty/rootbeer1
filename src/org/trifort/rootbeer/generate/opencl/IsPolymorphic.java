/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soot.FastHierarchy;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Type;
import soot.jimple.SpecialInvokeExpr;
import soot.rtaclassload.MethodSignature;
import soot.rtaclassload.MethodSignatureUtil;
import soot.rtaclassload.RTAClassLoader;

public class IsPolymorphic {
  
  private Type highestType;
  
  public IsPolymorphic(){
  }
  
  public boolean test(SootMethod soot_method){
    return test(soot_method, false);
  }
  
  public boolean test(SootMethod sootMethod, boolean specialInvoke){
    SootClass sootClass = sootMethod.getDeclaringClass();
    if(sootMethod.isConstructor() || specialInvoke || sootMethod.isNative()){
      highestType = sootClass.getType();
      return false;
    } else {
      List<MethodSignature> methods = OpenCLScene.v().getVirtualMethods(sootMethod.getSignature());
      if(methods.size() == 0){
        System.out.println("empty methods: "+sootMethod.getSignature());
      }
      findHighestType(methods);
      if(methods.size() == 1){
        return false;
      } else {
        return true;
      }
    }
  }

  private void findHighestType(List<MethodSignature> methods) {
    MethodSignature highest = methods.get(0);
    highestType = highest.getClassName().toSootType();
  }

  public Type getHighestType() {
    return highestType;
  }
}
