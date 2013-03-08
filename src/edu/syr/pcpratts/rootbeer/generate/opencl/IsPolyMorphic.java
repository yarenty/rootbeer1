/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import soot.*;
import soot.rbclassload.MethodSignatureUtil;

public class IsPolyMorphic {

  public boolean isPoly(SootMethod soot_method, List<Type> hierarchy){ 
    String name = soot_method.getName();
    if(name.equals("<init>") || name.equals("<clinit>")){
      return false;
    }
    if(soot_method.isAbstract()){
      return true;
    }
    MethodSignatureUtil util = new MethodSignatureUtil();
    util.parse(soot_method.getSignature());
    List<Type> params = util.getParameterTypesTyped();
    
    int count = 0;
    for(Type type : hierarchy){
      if(type instanceof RefType == false){
        continue;
      }
      RefType ref_type = (RefType) type;
      SootClass curr = ref_type.getSootClass();
      if(curr.declaresMethod(name, params)){
        count++;
      }
    }
    if(count >= 2){
      return true;
    }
    return false;
  }
}
