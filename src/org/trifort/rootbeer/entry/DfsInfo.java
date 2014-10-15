package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soot.ArrayType;
import soot.RefType;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Type;
import soot.rtaclassload.NumberedType;
import soot.rtaclassload.RTAClassLoader;
import soot.rtaclassload.RTAType;

public class DfsInfo {

  public static void reset(){
    instance = null;
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
  private Set<Type> newInvokes;
  private Set<Type> dfsTypes;
  private Set<Type> virtualMethodBases;
  private List<Type> orderedRefLikeTypes;
  private List<RefType> orderedRefTypes;
  
  public DfsInfo(){
    classes = new HashSet<SootClass>();
    methods = new HashSet<SootMethod>();
    fields = new HashSet<SootField>();
    instanceOfs = new HashSet<Type>();
    arrayTypes = new HashSet<ArrayType>();
    dfsTypes = new HashSet<Type>();
    newInvokes = new HashSet<Type>();
    orderedRefLikeTypes = new ArrayList<Type>();
    orderedRefTypes = new ArrayList<RefType>();
  }

  public void addClass(SootClass sootClass) {
    classes.add(sootClass);
    addType(sootClass.getType());
  }
  
  public void addMethod(SootMethod sootMethod) {
    methods.add(sootMethod);
    addType(sootMethod.getDeclaringClass().getType());
  }

  public void addField(SootField sootField) {
    fields.add(sootField);
    addType(sootField.getDeclaringClass().getType());
  }

  public void addInstanceOf(Type type) {
    instanceOfs.add(type);
    addType(type);
  }

  public void addType(Type type){
    dfsTypes.add(type);
    if(type instanceof RefType){
      RefType refType = (RefType) type;
      SootClass sootClass = refType.getSootClass();
      if(sootClass.hasSuperclass()){
        addType(sootClass.getSuperclass().getType());
      }
    }
  }
  
  public void addArrayType(ArrayType arrayType) {
    arrayTypes.add(arrayType);
    addType(arrayType);
    addType(arrayType.baseType);
  }
  
  public void addNewInvoke(Type type) {
    newInvokes.add(type);
    addType(type);
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
    Set<ArrayType> newArrayTypes = new HashSet<ArrayType>();
    for(ArrayType arrayType : arrayTypes){
      for(int dim = arrayType.numDimensions; dim > 0; --dim){
        ArrayType newArrayType = ArrayType.v(arrayType.baseType, dim);
        newArrayTypes.add(newArrayType);
      }
    }
    arrayTypes = newArrayTypes;
  }

  public void finalizeTypes() {
    List<NumberedType> numberedTypes = RTAClassLoader.v().getNumberedTypes();
    for(int i = numberedTypes.size() - 1; i >= 0; --i){
      NumberedType numberedType = numberedTypes.get(i);
      Type type = numberedType.getType().toSootType();
      if(dfsTypes.contains(type)){
        if(type instanceof RefType){
          RefType refType = (RefType) type;
          SootClass sootClass = refType.getSootClass();
          if(sootClass.isInterface() == false){
            orderedRefLikeTypes.add(type);
            orderedRefTypes.add(refType);
          }
        }
      }
    }
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

  public Set<Type> getNewInvokes() {
    return newInvokes;
  }

  public void setVirtualMethodBases(Set<Type> newInvokes) {
    virtualMethodBases = newInvokes;
  }
  
  public Set<Type> getVirtualMethodBases(){
    return virtualMethodBases;
  }
}
