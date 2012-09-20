CC  = mpicc
CXX = mpicxx
AR  = ar

metis_Dir = ./pkg/metis
metis_Include = -I${metis_Dir}/include
metis_lib = -L${metis_Dir}/lib

# CFLAGS = -c -DNDEBUG -Wall 
CFLAGS = -c -g -Wall
CPPFLAGS = -Iinclude
VPATH = src test

test.exe: pre test.o libhiprop.a
	$(CC) -g -o $@ $(CPPFLAGS) test.o -L./ -lhiprop
test2.exe: pre test2.o libhiprop.a
	$(CC) -g -o $@ $(CPPFLAGS) test2.o -L./ -lhiprop $(metis_Include) $(metis_lib) -lmetis
test3.exe: pre test3.o libhiprop.a
	$(CC) -g -o $@ $(CPPFLAGS) test3.o -L./ -lhiprop $(metis_Include) $(metis_lib) -lmetis

all: test.exe test2.exe test3.exe doc

lib: pre libhiprop.a

pre: ./include/stdafx.h.gch

./include/stdafx.h.gch: ./include/stdafx.h
	 $(CC) $(CFLAGS) $(CPPFLAGS) $<

doc:
	doxygen hiprop-doxygen-file

libhiprop.a: util.o hiprop.o
	$(AR) cru libhiprop.a $^
	ranlib libhiprop.a

%.o:%.c 
	$(CC) $(CFLAGS) $(CPPFLAGS) $(metis_Include) $(metis_lib) $< -o $@

%.o:%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $< -o $@  

%.o:%.cxx
	$(CXX) $(CFLAGS) $(CPPFLAGS) $< -o $@

tagsfile:
	ctags src/*.c include/*.h test/*.c
clean:
	rm -f *.o *.exe *.a
	rm -f *.gch
	rm -rf doc
	rm -f tags
	rm -f src/tags
	rm -f include/tags
	rm -f include/stdafx.h.gch
	rm -f test/tags

