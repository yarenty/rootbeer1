/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.classloader;

import edu.syr.pcpratts.rootbeer.generate.bytecode.MultiDimensionalArrayTypeCreator;
import java.util.*;
import soot.*;
import soot.jimple.toolkits.callgraph.CallGraph;

public class DfsInfo {

  private Set<String> m_dfsMethods;
  private Set<Type> m_dfsTypes;
  private CallGraph m_callGraph;
  private Map<Type, List<Type>> m_parentsToChildren;
  private List<NumberedType> m_numberedTypes;
  private List<Type> m_orderedTypes;
  private List<RefType> m_orderedRefTypes;
  private Set<SootField> m_dfsFields;
  
  public DfsInfo() {
    m_dfsMethods = new HashSet<String>();
    m_dfsTypes = new HashSet<Type>();
    m_dfsFields = new HashSet<SootField>();
    m_callGraph = new CallGraph();
    m_parentsToChildren = new HashMap<Type, List<Type>>();
  }
  
  public void expandArrayTypes(){
    MultiDimensionalArrayTypeCreator creator = new MultiDimensionalArrayTypeCreator();
    Set<Type> added = creator.create(m_dfsTypes);
    m_dfsTypes.addAll(added);
    SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
    for(Type added_type : added){
      addSuperClass(added_type, obj_class.getType());
    }
  }
  
  public void orderTypes(){
    List<NumberedType> numbered_types = new ArrayList<NumberedType>();
    int number = 1;
    SootClass obj_cls = Scene.v().getSootClass("java.lang.Object");
    Type curr = obj_cls.getType();
    List<Type> queue = new LinkedList<Type>();
    queue.add(curr);
    while(queue.isEmpty() == false){
      curr = queue.get(0);
      queue.remove(0);
      
      NumberedType numbered_type = new NumberedType(curr, number);
      numbered_types.add(numbered_type);
      number++;
      
      if(m_parentsToChildren.containsKey(curr) == false){
        continue;
      }
      
      List<Type> children = m_parentsToChildren.get(curr);
      queue.addAll(children);
    }
    
    m_numberedTypes = new ArrayList<NumberedType>();
    m_orderedTypes = new ArrayList<Type>();
    m_orderedRefTypes = new ArrayList<RefType>();
    for(int i = numbered_types.size() - 1; i >= 0; --i){
      NumberedType curr2 = numbered_types.get(i);
      m_numberedTypes.add(curr2);
      m_orderedTypes.add(curr2.getType());
      
      Type type = curr2.getType();
      if(type instanceof RefType){
        RefType ref_type = (RefType) type;
        m_orderedRefTypes.add(ref_type);
      }
    }
  }
  
  public List<NumberedType> getNumberedTypes(){
    return m_numberedTypes;
  }

  public void print() {
    printSet("methods: ", m_dfsMethods);
    System.out.println("parentsToChildren: ");
    for(Type parent : m_parentsToChildren.keySet()){
      List<Type> children = m_parentsToChildren.get(parent);
      System.out.println("  "+parent);
      for(Type child : children){
        System.out.println("    "+child);
      }
    }
  }

  private void printSet(String name, Set<String> curr_set) {
    System.out.println(name);
    for(String curr : curr_set){
      System.out.println("  "+curr);
    }
  }

  public List<String> getForwardReachables() {
    List<String> ret = new ArrayList<String>();
    ret.addAll(m_dfsMethods);
    return ret;
  }

  public void addMethod(String signature) {
    if(m_dfsMethods.contains(signature) == false){
      m_dfsMethods.add(signature);
    }
  }

  public boolean containsMethod(String signature) {
    return m_dfsMethods.contains(signature);
  }

  public boolean containsType(Type name) {
    return m_dfsTypes.contains(name);
  }

  public void addType(Type name) {
    if(m_dfsTypes.contains(name) == false){
      m_dfsTypes.add(name);
    }
  }
  
  public void addField(SootField field){
    if(m_dfsFields.contains(field) == false){
      m_dfsFields.add(field);
    }
  }

  public void addSuperClass(Type curr, Type superclass) {
    if(m_parentsToChildren.containsKey(superclass)){
      List<Type> children = m_parentsToChildren.get(superclass);
      if(children.contains(curr) == false){
        children.add(curr);
      }
    } else {
      List<Type> children = new ArrayList<Type>();
      children.add(curr);
      m_parentsToChildren.put(superclass, children);
    }
  }

  public List<Type> getOrderedTypes() {
    return m_orderedTypes;
  }

  public List<RefType> getOrderedRefTypes() {
    return m_orderedRefTypes;
  }

  public Set<String> getMethods() {
    return m_dfsMethods;
  }

  public Set<SootField> getFields() {
    return m_dfsFields;
  }
}
