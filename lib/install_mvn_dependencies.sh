#!/bin/sh

mvn install:install-file -Dfile=pack.jar -DgroupId=rootbeer \
    -DartifactId=pack -Dversion=1.0 -Dpackaging=jar

mvn install:install-file -Dfile=sootclasses-rbclassload.jar -DgroupId=soot \
    -DartifactId=soot -Dversion=rb-1.0 -Dpackaging=jar 

mvn install:install-file -Dfile=jasminclasses-2.5.0.jar -DgroupId=soot \
    -DartifactId=jasmin -Dversion=2.5.0 -Dpackaging=jar

mvn install:install-file -Dfile=polyglotclasses-1.3.5.jar -DgroupId=soot \
    -DartifactId=polyglot -Dversion=1.3.5 -Dpackaging=jar

mvn install:install-file -Dfile=AXMLPrinter2.jar -DgroupId=soot \
    -DartifactId=axmlprinter -Dversion=2.0 -Dpackaging=jar
