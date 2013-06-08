/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import soot.AnySubType;
import soot.FastHierarchy;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Type;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.HierarchyGraph;
import soot.rbclassload.MethodEqual;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;

/**
 * Represents all the versions of methods in a class Hierarchy
 * @author pcpratts
 */
public class MethodHierarchies {

  private Set<MethodHierarchy> m_hierarchies;
  private MethodSignatureUtil m_util;
  
  public MethodHierarchies(){
    m_hierarchies = new LinkedHashSet<MethodHierarchy>();
    m_util = new MethodSignatureUtil();
  }
  
  public void addMethod(SootMethod method){
    MethodHierarchy new_hierarchy = new MethodHierarchy(method);
    if(m_hierarchies.contains(new_hierarchy) == false)
      m_hierarchies.add(new_hierarchy);
  }
  
  public List<OpenCLMethod> getMethods(){
    List<OpenCLMethod> ret = new ArrayList<OpenCLMethod>();
    //for each method    
    for(MethodHierarchy method_hierarchy : m_hierarchies){
      //get the list of classes in the hierarchy
      List<OpenCLMethod> methods = method_hierarchy.getMethods();
      for(OpenCLMethod method : methods){ 
        ret.add(method);
      }
    }   
    return ret;
  }
  
  public List<OpenCLPolymorphicMethod> getPolyMorphicMethods(){
    List<OpenCLPolymorphicMethod> ret = new ArrayList<OpenCLPolymorphicMethod>();
    //for each method    
    for(MethodHierarchy method_hierarchy : m_hierarchies){
      if(method_hierarchy.isPolyMorphic()){
        ret.add(method_hierarchy.getOpenCLPolyMorphicMethod());
      }
    }   
    return ret;
  }
  
  private class MethodHierarchy {
    
    private String m_methodSubsignature;
    private SootMethod m_sootMethod;
    private HierarchyGraph m_hierarchyGraph;
    
    public MethodHierarchy(SootMethod method){
      m_methodSubsignature = method.getSubSignature();
      m_sootMethod = method;
      ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
      m_hierarchyGraph = class_hierarchy.getHierarchyGraph(m_sootMethod.getSignature());
    }
    
    public List<OpenCLMethod> getMethods(){
      List<OpenCLMethod> ret = new ArrayList<OpenCLMethod>();
      if(m_sootMethod.isConstructor()){
        OpenCLMethod method = new OpenCLMethod(m_sootMethod, m_sootMethod.getDeclaringClass());
        ret.add(method);
        return ret;
      }
      
      List<String> methods = RootbeerClassLoader.v().getClassHierarchy().getVirtualMethods(m_sootMethod.getSignature());
      for(String virt_method : methods){
        m_util.parse(virt_method);
        SootMethod soot_method = m_util.getSootMethod();
        OpenCLMethod method = new OpenCLMethod(soot_method, soot_method.getDeclaringClass());
        ret.add(method);
      }
      return ret;
    }
        
    public boolean isPolyMorphic(){
      IsPolymorphic poly_checker = new IsPolymorphic();
      if(poly_checker.test(m_sootMethod)){
        return true;
      }
      return false;
    }
    
    public OpenCLPolymorphicMethod getOpenCLPolyMorphicMethod(){
      SootClass soot_class = m_sootMethod.getDeclaringClass();
      List<SootClass> interfaces = new ArrayList<SootClass>();
      interfaces.addAll(soot_class.getInterfaces());
      String sub_sig = m_sootMethod.getSubSignature();
      for(SootClass iface : interfaces){
        if(iface.declaresMethod(sub_sig)){
          SootMethod iface_method = iface.getMethod(sub_sig);
          return new OpenCLPolymorphicMethod(iface_method);
        }
      }
      return new OpenCLPolymorphicMethod(m_sootMethod);
    }
    
    @Override
    public boolean equals(Object o){
      if(o instanceof MethodHierarchy == false)
        return false;
      MethodHierarchy other = (MethodHierarchy) o;
      if(m_methodSubsignature.equals(other.m_methodSubsignature) == false)
        return false;
      if(m_hierarchyGraph == other.m_hierarchyGraph == false){
        return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int hash = 7;
      hash = 59 * hash + (this.m_methodSubsignature != null ? this.m_methodSubsignature.hashCode() : 0);
      hash = 59 * hash + (this.m_hierarchyGraph != null ? this.m_hierarchyGraph.hashCode() : 0);
      return hash;
    }
    
    @Override
    public String toString(){
      return m_sootMethod.getSignature();
    }
  }
}
