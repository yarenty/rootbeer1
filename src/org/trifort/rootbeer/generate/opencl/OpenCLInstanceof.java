/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.trifort.rootbeer.entry.DfsInfo;
import org.trifort.rootbeer.generate.opencl.tweaks.Tweaks;

import soot.FastHierarchy;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.Type;
import soot.jimple.InstanceOfExpr;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;

public class OpenCLInstanceof {

  private Type m_type;
  private OpenCLType m_oclType;
  
  public OpenCLInstanceof(Type type) {
    m_type = type;
    m_oclType = new OpenCLType(m_type);
  }

  public String getPrototype() {
    return getDecl()+";\n";
  }
  
  private String getDecl(){
    String device = Tweaks.v().getDeviceFunctionQualifier();
    String global = Tweaks.v().getGlobalAddressSpaceQualifier();
    
    String ret = device+" char "+getMethodName();
    ret += "(int thisref, int * exception)";
    return ret;
  }
  
  private String getMethodName(){
    return "org_trifort_rootbeer_instanceof_"+m_oclType.getDerefString();
  }

  public String getBody() {
    if(m_type instanceof RefType == false){
      throw new RuntimeException("not supported yet");
    }
    RefType ref_type = (RefType) m_type;    
    List<Integer> type_list = getTypeList(ref_type);
    
    String ret = getDecl();
    ret += "{\n";
    ret += "  char * thisref_deref;\n";
    ret += "  GC_OBJ_TYPE_TYPE type;\n";
    ret += "  if(thisref == -1){\n";
    ret += "    return 0;\n";
    ret += "  }\n";
    ret += "  thisref_deref = org_trifort_gc_deref(thisref);\n";
    ret += "  type = org_trifort_gc_get_type(thisref_deref);\n";
    ret += "  switch(type){\n";
    for(Integer ntype : type_list){
      ret += "    case "+ntype+":\n";
    }
    ret += "      return 1;\n";
    ret += "  }\n";
    ret += "  return 0;\n";
    ret += "}\n";
    return ret;
  }
  
  public String invokeExpr(InstanceOfExpr arg0){
    String ret = getMethodName();
    ret += "("+arg0.getOp().toString()+", exception)";
    return ret;
  }
  
  @Override
  public boolean equals(Object other){
    if(other == null){
      return false;
    }
    if(other instanceof OpenCLInstanceof){
      OpenCLInstanceof rhs = (OpenCLInstanceof) other;
      return m_type.equals(rhs.m_type);
    } else {
      return false;
    }
    }

  @Override
  public int hashCode() {
    int hash = 5;
    hash = 29 * hash + (this.m_type != null ? this.m_type.hashCode() : 0);
    return hash;
  }

  private List<Integer> getTypeList(Type type) {
    Set<Type> visited = new TreeSet<Type>();
    LinkedList<Type> queue = new LinkedList<Type>();
    queue.add(type);
    
    Set<Type> newInvokes = DfsInfo.v().getNewInvokes();
    List<Integer> ret = new ArrayList<Integer>();
    
    while(queue.isEmpty() == false){
      Type entry = queue.removeFirst();
      if(visited.contains(entry)){
        continue;
      }
      visited.add(entry);
      
      if(type instanceof RefType){
        RefType refType = (RefType) type;
        SootClass sootClass = refType.getSootClass();
        if(newInvokes.contains(type)){
          ret.add(OpenCLScene.v().getTypeNumber(sootClass.getType()));
        }
        
        FastHierarchy hierarchy = Scene.v().getOrMakeFastHierarchy();
        Collection<SootClass> children = hierarchy.getSubclassesOf(sootClass);
        for(SootClass child : children){
          queue.add(child.getType());
        }
      }
      
    }
    
    return ret;
  }
}