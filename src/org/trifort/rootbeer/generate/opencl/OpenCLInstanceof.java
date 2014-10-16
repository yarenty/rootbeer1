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

import soot.ArrayType;
import soot.FastHierarchy;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.Type;
import soot.jimple.InstanceOfExpr;
import soot.rtaclassload.NumberedType;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.StringNumbers;

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
      throw new RuntimeException("not supported yet: instanceof "+m_type.toString());
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
    List<Integer> ret = new ArrayList<Integer>();
    List<Type> numberedTypes = OpenCLScene.v().getNumberedTypes();
    for(int i = 0; i < numberedTypes.size(); ++i){
      int number = i;
      Type currType = numberedTypes.get(i);
      if(parent(type, currType) || parent(currType, type)){
        ret.add(number);
      }
    }
    return ret;
  }

  private boolean parent(Type type0, Type type1) {
    if(type1 instanceof RefType){
      RefType refType1 = (RefType) type1;
      SootClass sootClass1 = refType1.getSootClass();
      if(type0 instanceof RefType){
        RefType refType0 = (RefType) type0;
        SootClass sootClass2 = refType0.getSootClass();
        LinkedList<SootClass> queue = new LinkedList<SootClass>();
        queue.add(sootClass2);
        while(queue.isEmpty() == false){
          SootClass curr = queue.removeFirst();
          if(curr.equals(sootClass1)){
            return true;
          } else {
            if(curr.hasSuperclass() == false){
              return false;
            } else {
              queue.add(curr.getSuperclass());
            }
            for(SootClass iface : curr.getInterfaces()){
              queue.add(iface);
            }
          }
        }
        return false;
      } else if(type0 instanceof ArrayType){
        if(sootClass1.getName().equals("java.lang.Object")){
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
}