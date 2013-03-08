/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import soot.*;
import soot.rbclassload.RootbeerClassLoader;

/**
 * Represents an OpenCL function that dispatches to the real OpenCL function
 * implementing the behavior of a certain classes version of a virtual method.
 * @author pcpratts
 */
public class OpenCLPolymorphicMethod {
  private final SootMethod m_sootMethod;

  //for hashcode
  private List<Type> m_hierarchy;

  public OpenCLPolymorphicMethod(SootMethod soot_method){
    m_sootMethod = soot_method;
  }

  public String getMethodPrototypes(){
    if(m_sootMethod.getName().equals("<init>"))
      return "";
    List<String> decls = getMethodDecls();
    StringBuilder ret = new StringBuilder();
    for(String decl : decls){
      decl += ";\n";
      ret.append(decl);
    }
    return ret.toString();
  }

  private List<String> getMethodDecls(){
    List<Type> hierarchy = getHierarchy();
    List<String> ret = new ArrayList<String>();
    for(Type type : hierarchy){
      if(type instanceof RefType == false){
        continue;
      }
      RefType ref_type = (RefType) type;
      SootClass soot_class = ref_type.getSootClass();
      OpenCLMethod ocl_method = new OpenCLMethod(m_sootMethod, soot_class);

      StringBuilder builder = new StringBuilder();
      builder.append(Tweaks.v().getDeviceFunctionQualifier()+" ");
      builder.append(ocl_method.getReturnString());
      builder.append(" invoke_"+ocl_method.getPolymorphicName());
      builder.append(ocl_method.getArgumentListStringPolymorphic());
      ret.add(builder.toString());
    }
    return ret;
  }

  public String getMethodBodies(){
    if(m_sootMethod.getName().equals("<init>"))
      return "";
    if(m_sootMethod.isConcrete() == false){
      return "";
    }
    List<String> decls = getMethodDecls();
    StringBuilder ret = new StringBuilder();
    for(String decl : decls){
      ret.append(getMethodBody(decl));
    }
    return ret.toString();
  }
  
  public String getMethodBody(String decl){
    List<Type> hierarchy = getHierarchy();
    StringBuilder ret = new StringBuilder();
    String address_qual = Tweaks.v().getGlobalAddressSpaceQualifier();
    //write function signature
    ret.append(decl);
    ret.append("{\n");

    if(m_sootMethod.isStatic()){
      if(m_sootMethod.getReturnType() instanceof VoidType == false){
        ret.append("return ");
      }
      Type first_type = hierarchy.get(0);
      RefType ref_type = (RefType) first_type;
      SootClass first_class = ref_type.getSootClass();
      String invoke_string = getStaticInvokeString(first_class);
      ret.append(invoke_string+"\n");
    } else {
      ret.append(address_qual+" char * thisref_deref;\n");
      ret.append("GC_OBJ_TYPE_TYPE derived_type;\n");
      ret.append("if(thisref == -1){\n");
      ret.append("  *exception = -2;\n");
      ret.append("return ");
      if(m_sootMethod.getReturnType() instanceof VoidType == false)
        ret.append("-1");
      ret.append(";\n");
      ret.append("}\n");
      ret.append("thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);\n");
      if(sizeHierarchy(hierarchy) == 1){
        SootClass sclass = getSingleMethodInHierarchy(hierarchy);
        String invoke_string = getInvokeString(sclass);
        if(m_sootMethod.getReturnType() instanceof VoidType == false){
          ret.append("return ");
        }
        ret.append(invoke_string+"\n");
      } else {
        ret.append("derived_type = edu_syr_pcpratts_gc_get_type(thisref_deref);\n");
        ret.append("if(0){}\n");
        int count = 0;
        for(Type type : hierarchy){
          if(type instanceof RefType == false){
            continue;
          }
          RefType curr_ref_type = (RefType) type;
          SootClass sclass = curr_ref_type.getSootClass();
          if(sootClassHasMethod(sclass) == false){
            continue;
          }
          ret.append("else if(derived_type == "+RootbeerClassLoader.v().getDfsInfo().getClassNumber(sclass)+"){\n");
          if(m_sootMethod.getReturnType() instanceof VoidType == false){
            ret.append("return ");
          }
          String invoke_string = getInvokeString(sclass);
          ret.append(invoke_string+"\n");
          ret.append("}\n");
          count++;
        }
      }
    }
    ret.append("return ");
    if(m_sootMethod.getReturnType() instanceof VoidType == false)
      ret.append("-1");
    ret.append(";\n");
    ret.append("}\n");
    return ret.toString();
  }

  //used to invoke polymorphic method inside this function
  private String getInvokeString(SootClass start_class){
    if(m_sootMethod.getName().equals("<init>"))
      return "";
    
    SootClass soot_class = start_class;
    SootMethod soot_method = null;
    String subsig = m_sootMethod.getSubSignature();
    while(true){
      if(soot_class.declaresMethod(subsig)){
        SootMethod curr = soot_class.getMethod(subsig);
        if(curr.isConcrete()){
          soot_method = curr;
          break;
        }
      }
      if(soot_class.hasSuperclass()){
        soot_class = soot_class.getSuperclass();
      } else {
        throw new RuntimeException("cannot find concrete base method: "+m_sootMethod.getSignature()+" "+start_class);
      }
    }
    
    OpenCLMethod ocl_method = new OpenCLMethod(soot_method, soot_class);
    String ret = ocl_method.getPolymorphicName() + "(";

    //write the gc_info and thisref
    ret += "gc_info, thisref";
    List args = soot_method.getParameterTypes();
    if(args.size() != 0)
      ret += ", ";

    for(int i = 0; i < args.size(); ++i){
      ret += "parameter" + Integer.toString(i);
      if(i < args.size() - 1)
        ret += ", ";
    }
    ret += ", exception);";
    return ret;
  }

  private String getStaticInvokeString(SootClass soot_class) {
    if(m_sootMethod.getName().equals("<init>"))
      return "";
    OpenCLMethod ocl_method = new OpenCLMethod(m_sootMethod, soot_class);
    String ret = ocl_method.getPolymorphicName() + "(";

    //write the gc_info and thisref
    ret += "gc_info";
    List args = m_sootMethod.getParameterTypes();
    if(args.size() != 0)
      ret += ", ";

    for(int i = 0; i < args.size(); ++i){
      ret += "parameter" + Integer.toString(i);
      if(i < args.size() - 1)
        ret += ", ";
    }
    ret += ", exception);";
    return ret;
  }

  private List<Type> getHierarchy(){
    SootClass soot_class = m_sootMethod.getDeclaringClass();
    List<Type> types = RootbeerClassLoader.v().getDfsInfo().getHierarchy(soot_class);
    List<Type> ret = new ArrayList<Type>();
    for(Type type : types){
      if(ret.contains(type) == false){
        ret.add(type);
      }
    }
    return ret;
  }

  @Override
  public boolean equals(Object o){
    if(o instanceof OpenCLPolymorphicMethod == false)
      return false;
    OpenCLPolymorphicMethod other = (OpenCLPolymorphicMethod) o;
    if(m_sootMethod.getName().equals(other.m_sootMethod.getName()) == false)
      return false;    
    if(getHierarchy().equals(other.getHierarchy()))
      return true;
    return false;
  }

  @Override
  public int hashCode() {
    m_hierarchy = getHierarchy();
    int hash = 5;
    hash = 53 * hash + (this.m_sootMethod != null ? this.m_sootMethod.hashCode() : 0);
    hash = 53 * hash + (this.m_hierarchy != null ? this.m_hierarchy.hashCode() : 0);
    return hash;
  }

  private boolean sootClassHasMethod(SootClass sclass) {
    String subsig = m_sootMethod.getSubSignature();
    return sootClassHasMethod0(sclass, subsig);
  }
  
  private boolean sootClassHasMethod0(SootClass sclass, String subsig){
    if(sclass.declaresMethod(subsig)){
      SootMethod soot_method = sclass.getMethod(subsig);
      if(soot_method.isConcrete()){
        return true;
      }
    }
    if(sclass.hasSuperclass()){
      return sootClassHasMethod0(sclass.getSuperclass(), subsig);
    } else {
      return false;
    }
  }

  private int sizeHierarchy(List<Type> hierarchy) {
    int ret = 0;
    for(Type type : hierarchy){
      if(type instanceof RefType == false){
        continue;
      }
      RefType ref_type = (RefType) type;
      SootClass sclass = ref_type.getSootClass();
      if(sootClassHasMethod(sclass) == false)
        continue;
      ret++;
    }
    return ret;
  }

  private SootClass getSingleMethodInHierarchy(List<Type> hierarchy) {
    for(Type type : hierarchy){
      if(type instanceof RefType == false){
        continue;
      }
      RefType ref_type = (RefType) type;
      SootClass soot_class = ref_type.getSootClass();
      if(sootClassHasMethod(soot_class) == false)
        continue;
      return soot_class;
    }
    return null;
  }

}
