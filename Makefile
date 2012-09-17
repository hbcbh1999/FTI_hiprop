CC  = mpicc
CXX = mpicxx
AR  = ar


# CFLAGS = -c -DNDEBUG -Wall 
CFLAGS = -c -H -g -Wall
CPPFLAGS = -Iinclude
VPATH = src test

test.exe: pre test.o libhiprop.a
	$(CXX) -o $@ $(CPPFLAGS) test.o -L./ -lhiprop
test2.exe: pre test2.o libhiprop.a
	$(CXX) -o $@ $(CPPFLAGS) test2.o -L./ -lhiprop


all: test.exe test2.exe doc

lib: pre libhiprop.a

pre: ./include/stdafx.h.gch

./include/stdafx.h.gch: ./include/stdafx.h
	 $(CC) $(CFLAGS) $(CPPFLAGS) $<

doc:
	doxygen hiprop-doxygen-file

libhiprop.a: memutil.o commutil.o io.o
	$(AR) cru libhiprop.a $^
	ranlib libhiprop.a

%.o:%.c 
	$(CC) $(CFLAGS)  $(CPPFLAGS) $< -o $@

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

