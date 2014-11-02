package org.trifort.rootbeer.generate.opencl;

public class OpenCLNumberedType implements Comparable<OpenCLNumberedType> {

  private String name;
  private int number;
  
  public OpenCLNumberedType(String name, int number){
    this.name = name;
    this.number = number;
  }
  
  public String getName(){
    return name;
  }
  
  public int getNumber(){
    return number;
  }

  @Override
  public int compareTo(OpenCLNumberedType o) {
    return Integer.compare(number, o.number);
  }
}
