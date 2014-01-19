package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class AutoboxingRunOnGpu implements Kernel {

  private Double m_double_result;
  private Integer m_integer_result;
  public void gpuMethod() {
    m_double_result = returnDouble();
    m_integer_result = returnInteger();
  }

  private double returnDouble() {
    return 10;
  }
  
  private int returnInteger() {
    return 0; 
    // values between -128 and 0 will fail because of problems in
    // static_getter_java_lang_Integer$IntegerCache_high or
    // static_getter_java_lang_Integer$IntegerCache_cache
    /*
       if ( i0  <  -128   ) goto label0;
       $i1  = static_getter_java_lang_Integer$IntegerCache_high(gc_info, exception);
       if ( i0  >  $i1   ) goto label0;
       $r0  = static_getter_java_lang_Integer$IntegerCache_cache(gc_info, exception);
       $i2  =  i0  +  128  ;
       $r1  = java_lang_Integer__array_get(gc_info, $r0, $i2, exception);
       if(*exception != 0) {
       return 0; }
       return  $r1 ;
     */
  }
  
  public double getDoubleResult(){
    return m_double_result;
  }
  
  public double getIntegerResult(){
    return m_integer_result;
  }
  
  public boolean compare(AutoboxingRunOnGpu rhs) {
    try {
      if(getDoubleResult() != rhs.getDoubleResult()){
        System.out.println("m_double_result");
        System.out.println("lhs: "+getDoubleResult());
        System.out.println("rhs: "+rhs.getDoubleResult());
        return false;
      }
      if(getIntegerResult() != rhs.getIntegerResult()){
        System.out.println("m_integer_result");
        System.out.println("lhs: "+getIntegerResult());
        System.out.println("rhs: "+rhs.getIntegerResult());
        return false;
      }
      return true;
    } catch(Exception ex){
      System.out.println("exception thrown");
      return false;
    }
  }
  
}
