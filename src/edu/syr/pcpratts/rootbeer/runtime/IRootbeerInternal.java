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
  public void runAll(Kernel jobs);
  public void runAll(List<Kernel> jobs);
  public Iterator<Kernel> run(Iterator<Kernel> jobs);
  public void setThreadConfig(ThreadConfig thread_config);
  public void clearThreadConfig();
  public void printMem(int start, int len);
}
