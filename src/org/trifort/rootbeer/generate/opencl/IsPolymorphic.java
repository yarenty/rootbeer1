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
import soot.rbclassload.MethodSignature;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;

public class IsPolymorphic {
  
  private Type highestType;
  
  public IsPolymorphic(){
  }
  
  public boolean test(SootMethod soot_method){
    return test(soot_method, false);
  }
  
  public boolean test(SootMethod sootMethod, boolean specialInvoke){
    SootClass sootClass = sootMethod.getDeclaringClass();
    highestType = sootClass.getType();
    if(sootClass.isInterface()){
      return true;
    }
    
    FastHierarchy fastHierarchy = Scene.v().getOrMakeFastHierarchy();
    Set<SootMethod> methods = fastHierarchy.resolveAbstractDispatch(sootClass, sootMethod);
    
    if(methods.size() == 1 || sootMethod.isConstructor() || specialInvoke){
      return false;
    } else {
      findHighestType(methods);
      return true;
    }
  }

  private void findHighestType(Set<SootMethod> methods) {
    Map<SootClass, Boolean> parentExists = new HashMap<SootClass, Boolean>();
    for(SootMethod method : methods){
      SootClass declaring = method.getDeclaringClass();
      if(declaring.hasSuperclass() == false){
        highestType = declaring.getType();
        return;
      }
      parentExists.put(declaring, false);
    }
    for(SootMethod method : methods){
      SootClass declaring = method.getDeclaringClass();
      if(declaring.hasSuperclass()){
        SootClass parent = declaring.getSuperclass();
        parentExists.put(parent, true);
      }
    }
    for(SootClass key : parentExists.keySet()){
      if(parentExists.get(key) == false){
        highestType = key.getType();
        return;
      }
    }
    throw new RuntimeException("cannot find highest type");
  }

  public Type getHighestType() {
    return highestType;
  }
}
