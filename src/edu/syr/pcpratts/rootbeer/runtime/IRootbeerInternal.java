/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;
import java.util.List;

public interface IRootbeerInternal {

  void runAll(List<Kernel> jobs);
  Iterator<Kernel> run(Iterator<Kernel> jobs);
  void setThreadConfig(ThreadConfig thread_config);
  void clearThreadConfig();
}
