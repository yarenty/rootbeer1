package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;

public class PolymorphicRunOnGpu implements Kernel {

  private PolymorphicClassBase m_object;
  
  public PolymorphicRunOnGpu(){
    m_object = new PolymorphicClassBase();
  }
  
  @Override
  public void gpuMethod() {
    m_object = new PolymorphicClassDerived();
  }

  public boolean compare(PolymorphicRunOnGpu rhs) {
    if(m_object instanceof PolymorphicClassDerived == false){
      System.out.println("m_object type");
      return false;
    }
    if(rhs.m_object instanceof PolymorphicClassDerived == false){
      System.out.println("rhs.m_object type");
      return false;
    }
    return true;
  }

}
