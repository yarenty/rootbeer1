/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime.math;

public class LongDouble {
  
  private byte m_sign;
  private short m_exponent;
  private long m_fraction;
 
  /**
   * http://en.wikipedia.org/wiki/Double-precision_floating-point_format
   * value = (-1)^sign * (1 + sum[i=1..52]{b_(-i)*2^(-I)}) * 2*(e-1023)
   * @param value 
   */
  public LongDouble(double value){
    long bits = Double.doubleToLongBits(value);
    m_sign = (byte) (((byte) (bits >> 63)) & (byte) 0x1);
    m_exponent = (short) ((short)(bits >> 52) & (short) 0x7ff);
    m_exponent -= (short) 1023;
    m_fraction = bits & 0x2ffffff;
  }
  
  public void exp(double power){
    long bits = Double.doubleToLongBits(power);
    byte pow_sign = (byte) (((byte) (bits >> 63)) & (byte) 0x1);
    short pow_exponent = (short) ((short)(bits >> 52) & (short) 0x7ff);
    pow_exponent -= (short) 1023;
    long pow_fraction = (bits & 0x2ffffff) - 1023;
    LongDouble power_ld = new LongDouble(power);
    power_ld.print();
  }
  
  private void print(){
    System.out.println("print");
    System.out.println("  m_sign: "+m_sign);
    System.out.println("  m_exponenet: "+m_exponent);
    System.out.println("  m_fraction: "+m_fraction);
  }
  
  public static void main(String[] args){
    LongDouble ld = new LongDouble(1.0);
    ld.print();
    ld.exp(1020.0);
  }
}
