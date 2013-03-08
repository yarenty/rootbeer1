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
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;

/**
 * Represents all the versions of methods in a class Hierarchy
 * @author pcpratts
 */
public class MethodHierarchies {

  private Set<MethodHierarchy> m_hierarchies;
  
  public MethodHierarchies(){
    m_hierarchies = new LinkedHashSet<MethodHierarchy>();
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
    private List<Type> m_hierarchy;
    
    public MethodHierarchy(SootMethod method){
      m_methodSubsignature = method.getSubSignature();
      m_sootMethod = method;
    }
    
    public List<OpenCLMethod> getMethods(){
      List<OpenCLMethod> ret = new ArrayList<OpenCLMethod>();
      if(m_sootMethod.isConstructor()){
        OpenCLMethod method = new OpenCLMethod(m_sootMethod, m_sootMethod.getDeclaringClass());
        ret.add(method);
        return ret;
      }
      
      Set<Type> class_hierarchy = RootbeerClassLoader.v().getDfsInfo().getPointsTo(m_sootMethod.getSignature());
      Set<SootClass> valid_hierarchy_classes = RootbeerClassLoader.v().getValidHierarchyClasses();
      
      if(class_hierarchy == null){
        class_hierarchy = new HashSet<Type>();
        List<Type> class_hierarchy2 = RootbeerClassLoader.v().getDfsInfo().getHierarchy(m_sootMethod.getDeclaringClass());
        class_hierarchy.addAll(class_hierarchy2);
      }
      
      MethodSignatureUtil util = new MethodSignatureUtil();
      util.parse(m_sootMethod.getSignature());
      String method_name = util.getMethodName();
      List<Type> params = util.getParameterTypesTyped();
      
      for(Type type : class_hierarchy){
        if(type instanceof RefType){
          RefType ref_type = (RefType) type;
          SootClass soot_class = ref_type.getSootClass();
          if(soot_class.declaresMethod(method_name, params)){
            List<SootMethod> methods = soot_class.getMethods();
            List<SootMethod> found_methods = new ArrayList<SootMethod>();
            for(SootMethod method : methods){
              if(method.getName().equals(method_name) &&
                 method.getParameterCount() == params.size()){
                
                if(typesEqual(method.getParameterTypes(), params)){
                  found_methods.add(method);
                }
              }
            }
            
            if(found_methods.size() == 1){
              SootMethod soot_method = found_methods.get(0);
              if(soot_method.isConcrete() == false){
                continue;
              }
              OpenCLMethod method = new OpenCLMethod(soot_method, soot_class);
              ret.add(method);
            } else {
              //select method with same return type as declaring class
              int matching_count = 0;
              for(SootMethod soot_method : found_methods){
                if(soot_method.getReturnType().equals(RefType.v(m_sootMethod.getDeclaringClass()))){
                  if(soot_method.isConcrete() == false){
                    continue;
                  }
                  OpenCLMethod method = new OpenCLMethod(soot_method, soot_class);
                  ret.add(method);
                  matching_count++;
                }
              }
              if(matching_count == 0){
                throw new RuntimeException("matching_count == 0");
              }
            }
          }
        }
        if(type instanceof AnySubType){
          AnySubType any_sub_type = (AnySubType) type;
          RefType base = any_sub_type.getBase();
          SootClass soot_class = base.getSootClass();
          FastHierarchy fast_hierarchy = Scene.v().getOrMakeFastHierarchy();
          Collection<SootClass> subclasses_col = fast_hierarchy.getSubclassesOf(soot_class);
          List<SootClass> subclasses = new ArrayList<SootClass>();
          subclasses.addAll(subclasses_col);
          subclasses.add(soot_class);
          for(SootClass subclass : subclasses){
            if(valid_hierarchy_classes.contains(subclass) == false){
              continue;
            }
            if(subclass.declaresMethod(m_methodSubsignature)){
              SootMethod soot_method = subclass.getMethod(m_methodSubsignature);
              if(soot_method.isConcrete() == false){
                continue;
              }
              OpenCLMethod method = new OpenCLMethod(soot_method, subclass);
              ret.add(method);
            }
          }
        }
      }      
      return ret;
    }
        
    public boolean isPolyMorphic(){
      List<Type> class_hierarchy = RootbeerClassLoader.v().getDfsInfo().getHierarchy(m_sootMethod.getDeclaringClass());
      IsPolyMorphic poly_checker = new IsPolyMorphic();
      if(poly_checker.isPoly(m_sootMethod, class_hierarchy)){
        return true;
      }
      return false;
    }
    
    public OpenCLPolymorphicMethod getOpenCLPolyMorphicMethod(){
      List<Type> class_hierarchy = RootbeerClassLoader.v().getDfsInfo().getHierarchy(m_sootMethod.getDeclaringClass());
      for(Type type : class_hierarchy){
        if(type instanceof RefType == false){
          continue;
        }
        RefType ref_type = (RefType) type;
        SootClass soot_class = ref_type.getSootClass();
        try {
          SootMethod soot_method = soot_class.getMethod(m_methodSubsignature);
          return new OpenCLPolymorphicMethod(soot_method);
        } catch(RuntimeException ex){
          continue;
        }
      }
      throw new RuntimeException("Cannot find class: "+m_methodSubsignature);
    }
    
    @Override
    public boolean equals(Object o){
      if(o instanceof MethodHierarchy == false)
        return false;
      MethodHierarchy other = (MethodHierarchy) o;
      if(m_methodSubsignature.equals(other.m_methodSubsignature) == false)
        return false;
      saveHierarchy();
      other.saveHierarchy();
      if(m_hierarchy == other.m_hierarchy == false){
        return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int hash = 7;
      hash = 59 * hash + (this.m_methodSubsignature != null ? this.m_methodSubsignature.hashCode() : 0);
      hash = 59 * hash + (this.m_hierarchy != null ? this.m_hierarchy.hashCode() : 0);
      return hash;
    }

    private void saveHierarchy() {
      m_hierarchy = RootbeerClassLoader.v().getDfsInfo().getHierarchy(m_sootMethod.getDeclaringClass());
    }

    private boolean typesEqual(List<Type> types1, List<Type> types2) {
      for(int i = 0; i < types1.size(); ++i){
        Type lhs = types1.get(i);
        Type rhs = types2.get(i);
        if(lhs.equals(rhs) == false){
          return false;
        }
      }
      return true;
    }
  }
}
