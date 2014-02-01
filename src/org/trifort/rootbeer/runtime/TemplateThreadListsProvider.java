/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */
package org.trifort.rootbeer.runtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TemplateThreadListsProvider {

  private List<TemplateThread> sleeping = Collections.synchronizedList(new ArrayList<TemplateThread>());
  private List<TemplateThread> computing = Collections.synchronizedList(new ArrayList<TemplateThread>());

  public TemplateThreadListsProvider() {
    for (int i = 0; i < Runtime.getRuntime().availableProcessors(); ++i) {
      TemplateThread t = new TemplateThread(this);
      t.start();
      sleeping.add(t);
    }
  }
  
  public List<TemplateThread> getSleeping() {
    return sleeping;
  }

  public List<TemplateThread> getComputing() {
    return computing;
  }
}
