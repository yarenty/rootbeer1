package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class AutoboxingRunOnGpu implements Kernel {

  private Double m_result;
  public void gpuMethod() {
    m_result = returnDouble();
  }

  private double returnDouble() {
    return 10;
  }
  
  public double getResult(){
    return m_result;
  }

  public boolean compare(AutoboxingRunOnGpu rhs) {
    try {
      if(getResult() != rhs.getResult()){
        System.out.println("m_result");
        System.out.println("lhs: "+getResult());
        System.out.println("rhs: "+rhs.getResult());
        return false;
      }
      return true;
    } catch(Exception ex){
      System.out.println("exception thrown");
      return false;
    }
  }
  
}
