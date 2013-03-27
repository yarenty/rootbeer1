/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */
package edu.syr.pcpratts.rootbeer.runtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TemplateThreadListsProvider {

  private List<TemplateThread> sleeping = Collections.synchronizedList(new ArrayList<TemplateThread>());
  private List<TemplateThread> computing = Collections.synchronizedList(new ArrayList<TemplateThread>());
  private static TemplateThreadListsProvider instance = null;

  private TemplateThreadListsProvider() {
    for (int i = 0; i < Runtime.getRuntime().availableProcessors(); ++i) {
      TemplateThread t = new TemplateThread();
      t.start();
      sleeping.add(t);
    }
  }

  public static TemplateThreadListsProvider getInstance() {
    if (instance == null) {
      instance = new TemplateThreadListsProvider();
    }
    return instance;
  }

  public List<TemplateThread> getSleeping() {
    return sleeping;
  }

  public List<TemplateThread> getComputing() {
    return computing;
  }
}
