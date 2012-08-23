CC  = mpicc
CXX = mpicxx
AR  = ar

# CFLAGS = -c -DNDEBUG -Wall 
CFLAGS = -c -g -Wall
CPPFLAGS = -Iinclude
VPATH = src test


test.exe: test.o libhiprop.a
	$(CXX) -o $@ $(CPPFLAGS) test.o -L./ -lhiprop

all: test.exe doc

lib: libhiprop.a

doc:
	doxygen hiprop-doxygen-file

libhiprop.a: memutil.o
	$(AR) cru libhiprop.a $^
	ranlib libhiprop.a

%.o:%.c 
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

%.o:%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS) $< -o $@  

%.o:%.cxx
	$(CXX) $(CFLAGS) $(CPPFLAGS) $< -o $@

tagsfile:
	ctags src/*.c include/*.h test/*.c
	cp tags src/
	cp tags include/
	cp tags test/
clean:
	rm -f *.o *.exe *.a
	rm -rf doc
	rm -f tags
	rm -f src/tags
	rm -f include/tags
	rm -f test/tags

