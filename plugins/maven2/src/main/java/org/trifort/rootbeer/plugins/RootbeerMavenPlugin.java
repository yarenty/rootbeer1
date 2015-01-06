package org.trifort.rootbeer.plugins;

import org.apache.maven.model.Build;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.project.MavenProject;

@Mojo(name="rootbeer-maven-plugin")
public class RootbeerMavenPlugin extends AbstractMojo
{
  public void execute() throws MojoExecutionException {
    MavenProject project = (MavenProject) getPluginContext().get("project");
    Build build = project.getBuild();
    String outputDirectory = build.getOutputDirectory();
    getLog().info("Hello World: "+outputDirectory);
  }
}
