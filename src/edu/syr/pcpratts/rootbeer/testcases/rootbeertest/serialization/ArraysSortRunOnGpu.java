package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import java.util.Arrays;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class ArraysSortRunOnGpu implements Kernel {

  private int[] m_array;
  
  public ArraysSortRunOnGpu(){
    m_array = new int[8];
    for(int i = 0; i < m_array.length; ++i){
      m_array[i] = m_array.length - i;
    }
  }
  
	@Override
	public void gpuMethod() {
	  Arrays.sort(m_array);
	}
	
	private String toString(int[] array){
	  String ret = "[";
	  for(int i = 0; i < array.length; ++i){
	    ret += array[i];
	    if(i < array.length - 1){
	      ret += ",";
	    }
	  }
	  ret += "]";
	  return ret;
	}

  public boolean compare(ArraysSortRunOnGpu rhs) {
    if(m_array.length != rhs.m_array.length){
      System.out.println("m_array.length");
      return false;
    }
    for(int i = 0; i < m_array.length; ++i){
      int lhs_value = m_array[i];
      int rhs_value = rhs.m_array[i];
      if(lhs_value != rhs_value){
        System.out.println("value");
        System.out.println("lhs: "+toString(m_array));
        System.out.println("rhs: "+toString(rhs.m_array));
        return false;
      }
    }
    return true;
  }
}
