package org.trifort.rootbeer.entry;

import java.util.List;
import java.util.Set;

import soot.ArrayType;
import soot.RefType;
import soot.SootField;
import soot.Type;

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
  
  public DfsInfo(){
    
  }

  public void addType(int className) {
    // TODO Auto-generated method stub
    
  }

  public void addMethod(String string) {
    // TODO Auto-generated method stub
    
  }

  public void addField(SootField soot_field) {
    // TODO Auto-generated method stub
    
  }

  public void addInstanceOf(Type type) {
    // TODO Auto-generated method stub
    
  }

  public void addType(Type type) {
    // TODO Auto-generated method stub
    
  }

  public Set<Type> getDfsTypes() {
    // TODO Auto-generated method stub
    return null;
  }

  public List<Type> getOrderedRefLikeTypes() {
    // TODO Auto-generated method stub
    return null;
  }

  public List<RefType> getOrderedRefTypes() {
    // TODO Auto-generated method stub
    return null;
  }

  public Set<ArrayType> getArrayTypes() {
    // TODO Auto-generated method stub
    return null;
  }

  public void expandArrayTypes() {
    // TODO Auto-generated method stub
    
  }

  public void finalizeTypes() {
    // TODO Auto-generated method stub
    
  }

  public Set<String> getMethods() {
    // TODO Auto-generated method stub
    return null;
  }

  public Set<SootField> getFields() {
    // TODO Auto-generated method stub
    return null;
  }

  public Set<Type> getInstanceOfs() {
    // TODO Auto-generated method stub
    return null;
  }

  public Set<Integer> getNewInvokes() {
    // TODO Auto-generated method stub
    return null;
  }
}
