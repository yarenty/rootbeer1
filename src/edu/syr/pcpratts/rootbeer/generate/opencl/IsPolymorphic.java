/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.List;
import soot.SootMethod;
import soot.jimple.SpecialInvokeExpr;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.RootbeerClassLoader;

public class IsPolymorphic {
  
  private SootMethod m_baseMethod;
  
  public boolean test(SootMethod soot_method){
    return test(soot_method, false);
  }
  
  public boolean test(SootMethod soot_method, boolean special_invoke){
    String signature = soot_method.getSignature();
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    List<SootMethod> virtual_methods = class_hierarchy.getAllVirtualMethods(signature);
    
    m_baseMethod = virtual_methods.get(0);
    
    if(virtual_methods.size() == 1 || m_baseMethod.isConstructor() || special_invoke){
      return false;
    } else {
      return true;
    }
  }

  public SootMethod getBaseMethod() {
    return m_baseMethod;
  }
}
