/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.test;

import edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired.ChangeThreadTest;
import java.util.ArrayList;
import java.util.List;

public class ChangeThread implements TestSerializationFactory {

  public List<TestSerialization> getProviders() {
    List<TestSerialization> ret = new ArrayList<TestSerialization>();
    ret.add(new ChangeThreadTest());
    return ret;
  }

  public void makeHarder() {
    //ignore
  }

}
