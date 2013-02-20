/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.test;

import edu.syr.pcpratts.rootbeer.testcases.rootbeertest.apps.fastmatrixdebug.MatrixApp;
import java.util.ArrayList;
import java.util.List;

public class ApplicationMain implements TestApplicationFactory {

  public List<TestApplication> getProviders() {
    List<TestApplication> ret = new ArrayList<TestApplication>();
    ret.add(new MatrixApp());
    return ret;
  }

}
