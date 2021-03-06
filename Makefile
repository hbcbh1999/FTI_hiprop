CC  = mpicc
CXX = mpicxx
AR  = ar

# CFLAGS = -c -DNDEBUG -Wall 
CFLAGS = -g -Wall -c
Include_Dir = -Iinclude
CPPFLAGS = 
VPATH = src test

test.exe: pre test.o libhiprop.a
	$(CC) -g -o $@ test.o -lm -L./ -lhiprop -lrt
test2.exe: pre test2.o libhiprop.a
	$(CC) -g -o $@ test2.o -lm -L./ -lhiprop -lrt
test3.exe: pre test3.o libhiprop.a
	$(CC) -g -o $@ test3.o -lm -L./ -lhiprop -lrt

all: test.exe test2.exe test3.exe

lib: pre libhiprop.a

pre: ./include/stdafx.h.gch

./include/stdafx.h.gch: ./include/stdafx.h
	  $(CC) $(CFLAGS) $<

doc:
	doxygen hiprop-doxygen-file

libhiprop.a: nonfinite_util.o emx_util.o smoothing_clean.o compute_diffops_clean.o obtain_ringsz.o util.o hiprop.o
	$(AR) cru libhiprop.a $^
	ranlib libhiprop.a

%.o:%.c 
	$(CC) $(CFLAGS) $(Include_Dir) $< -o $@ -lm

%.o:%.cpp
	$(CXX) $(CPPFLAGS) $(Include_Dir)  $< -o $@ -lm

%.o:%.cxx
	$(CXX) $(CPPFLAGS) $(Include_Dir) $< -o $@ -lm

tagsfile:
	ctags -R --c++-kinds=+p --fields=+iaS --extra=+q src/*.c include/*.h test/*.c
clean:
	rm -f *.o *.exe *.a
	rm -f *.gch
	rm -rf doc
	rm -f include/stdafx.h.gch
	rm -f *.vtk
	rm -f run-log.*
	rm -f *.out

cleanout:
	rm -f *.vtk
	rm -f run-log.*
