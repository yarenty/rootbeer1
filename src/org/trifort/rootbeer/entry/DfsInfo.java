package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import soot.ArrayType;
import soot.RefType;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Type;
import soot.rbclassload.RTAType;

public class DfsInfo {

  public static void reset(){
    
  }
  
  public static DfsInfo v(){
    if(instance == null){
      instance = new DfsInfo();
    }
    return instance;
  }
  
  private static DfsInfo instance;

  private Set<SootClass> classes;
  private Set<SootMethod> methods;
  private Set<SootField> fields;
  private Set<Type> instanceOfs;
  private Set<ArrayType> arrayTypes;
  private Set<Type> dfsTypes;
  private List<Type> orderedRefLikeTypes;
  private List<RefType> orderedRefTypes;
  
  public DfsInfo(){
    classes = new HashSet<SootClass>();
    methods = new HashSet<SootMethod>();
    fields = new HashSet<SootField>();
    instanceOfs = new HashSet<Type>();
    arrayTypes = new HashSet<ArrayType>();
    dfsTypes = new HashSet<Type>();
    orderedRefLikeTypes = new ArrayList<Type>();
    orderedRefTypes = new ArrayList<RefType>();
  }

  public void addClass(SootClass sootClass) {
    classes.add(sootClass);
  }
  
  public void addMethod(SootMethod sootMethod) {
    methods.add(sootMethod);
  }

  public void addField(SootField sootField) {
    fields.add(sootField);
  }

  public void addInstanceOf(Type type) {
    instanceOfs.add(type);
  }

  public Set<Type> getDfsTypes() {
    return dfsTypes;
  }

  public List<Type> getOrderedRefLikeTypes() {
    return orderedRefLikeTypes;
  }

  public List<RefType> getOrderedRefTypes() {
    return orderedRefTypes;
  }

  public Set<ArrayType> getArrayTypes() {
    return arrayTypes;
  }

  public void expandArrayTypes() {
    // TODO Auto-generated method stub
    
  }

  public void finalizeTypes() {
    // TODO Auto-generated method stub
    
  }

  public Set<SootMethod> getMethods() {
    return methods;
  }

  public Set<SootField> getFields() {
    return fields;
  }

  public Set<Type> getInstanceOfs() {
    return instanceOfs;
  }

  public Set<Integer> getNewInvokes() {
    // TODO Auto-generated method stub
    return null;
  }
}
