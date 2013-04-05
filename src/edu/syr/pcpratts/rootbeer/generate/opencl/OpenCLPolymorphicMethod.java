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
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.HierarchyGraph;
import soot.rbclassload.RootbeerClassLoader;

/**
 * Represents an OpenCL function that dispatches to the real OpenCL function
 * implementing the behavior of a certain classes version of a virtual method.
 * @author pcpratts
 */
public class OpenCLPolymorphicMethod {
  private final SootMethod m_sootMethod;

  //for hashcode
  private HierarchyGraph m_hierarchyGraph;

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
    List<SootMethod> virtual_methods = getVirtualMethods();
    List<String> ret = new ArrayList<String>();
    for(SootMethod virtual_method : virtual_methods){
      SootClass soot_class = virtual_method.getDeclaringClass();
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
  
  private List<SootMethod> getVirtualMethods(){
    ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
    List<SootMethod> virtual_methods = class_hierarchy.getAllVirtualMethods(m_sootMethod.getSignature());
    return virtual_methods;
  }
  
  public String getMethodBody(String decl){
    StringBuilder ret = new StringBuilder();
    String address_qual = Tweaks.v().getGlobalAddressSpaceQualifier();
    //write function signature
    ret.append(decl);
    ret.append("{\n");
    
    List<SootMethod> virtual_methods = getVirtualMethods();
    if(m_sootMethod.isStatic()){
      if(m_sootMethod.getReturnType() instanceof VoidType == false){
        ret.append("return ");
      }
      Type first_type = m_sootMethod.getDeclaringClass().getType();
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
      if(virtual_methods.size() == 1){
        SootClass sclass = virtual_methods.get(0).getDeclaringClass();
        String invoke_string = getInvokeString(sclass);
        if(m_sootMethod.getReturnType() instanceof VoidType == false){
          ret.append("return ");
        }
        ret.append(invoke_string+"\n");
      } else {
        ret.append("derived_type = edu_syr_pcpratts_gc_get_type(thisref_deref);\n");
        ret.append("if(0){}\n");
        int count = 0;
        for(SootMethod method : virtual_methods){
          SootClass sclass = method.getDeclaringClass();
          ret.append("else if(derived_type == "+RootbeerClassLoader.v().getClassNumber(sclass)+"){\n");
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

  @Override
  public boolean equals(Object o){
    if(o instanceof OpenCLPolymorphicMethod == false)
      return false;
    OpenCLPolymorphicMethod other = (OpenCLPolymorphicMethod) o;
    if(m_sootMethod.getName().equals(other.m_sootMethod.getName()) == false)
      return false;    
    if(getHierarchyGraph().equals(other.getHierarchyGraph()))
      return true;
    return false;
  }

  @Override
  public int hashCode() {
    m_hierarchyGraph = getHierarchyGraph();
    int hash = 5;
    hash = 53 * hash + (this.m_sootMethod != null ? this.m_sootMethod.hashCode() : 0);
    hash = 53 * hash + (this.m_hierarchyGraph != null ? this.m_hierarchyGraph.hashCode() : 0);
    return hash;
  }

  private HierarchyGraph getHierarchyGraph(){
    if(m_hierarchyGraph == null){
      ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
      m_hierarchyGraph = class_hierarchy.getHierarchyGraph(m_sootMethod.getSignature());
    }
    return m_hierarchyGraph;
  }
}
