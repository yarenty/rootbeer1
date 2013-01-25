/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

class AbstractTestDerivedClass extends AbstractTestBaseClass {

  public AbstractTestDerivedClass() {
  }

  @Override
  public int add(int x, int y) {
    return x + y;
  }

}
