#include <dlfcn.h>
#include <fcntl.h>
#include <gelf.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "timing.h"

extern char* wrapper_funcname;

static const char* libName = "libenvyrt.so";
static void* libHandle = NULL;

#define WRAP_FUNCTION_BEGIN(name, retty, ...) \
retty name(__VA_ARGS__) \
{ \
	if (!libHandle) \
	{ \
		libHandle = dlopen(libName, RTLD_LAZY); \
		if (!libHandle) \
		{ \
			fprintf(stderr, "Cannot dlopen \"%s\"", libName); \
			exit(-1); \
		} \
	} \
	typedef retty(*name##_t)(__VA_ARGS__); \
	static name##_t __real_##name = NULL; \
	if (!__real_##name) \
	{ \
		__real_##name = dlsym(libHandle, "" #name ""); \
		if (!__real_##name) \
		{ \
			fprintf(stderr, "Cannot dlsym \"%s\"", "" #name ""); \
			exit(-1); \
		} \
	} \

#define WRAP_FUNCTION_END }

static char* elfGetSymData(const char* symname)
{
	char* result = NULL;
	const char* filename = "/proc/self/exe";

	int fd = -1;
	Elf *e = NULL;

	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		fprintf(stderr, "Cannot initialize ELF library: %s\n",
			elf_errmsg(-1));
		goto finish;
	}
	
	if ((fd = open(filename, O_RDONLY)) < 0) {
		fprintf(stderr, "Cannot open file %s\n", filename);
		goto finish;
	}
	if ((e = elf_begin(fd, ELF_C_READ, e)) == 0) {
		fprintf(stderr, "Cannot read ELF image from %s: %s\n",
			filename, elf_errmsg(-1));
		goto finish;
	}
	size_t shstrndx;
	if (elf_getshdrstrndx(e, &shstrndx)) {
		fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
			filename, elf_errmsg(-1));
		goto finish;
	}

	Elf_Scn* scn = elf_nextscn(e, NULL);
	int strndx;
	Elf_Data* symtab_data = NULL;
	for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
	{
		GElf_Shdr shdr;
		if (!gelf_getshdr(scn, &shdr)) {
			fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
				filename, elf_errmsg(-1));
			goto finish;
		}

		if (shdr.sh_type != SHT_SYMTAB) continue;

		symtab_data = elf_getdata(scn, NULL);
		if (!symtab_data)
		{
			fprintf(stderr, "Expected %s data section for %s\n",
				".symtab", filename);
			goto finish;
		}
		if (shdr.sh_size && !shdr.sh_entsize)
		{
			fprintf(stderr, "Cannot get the number of symbols for %s\n",
				filename);
			goto finish;
		}
		int nsymbols = 0;
		if (shdr.sh_size)
			nsymbols = shdr.sh_size / shdr.sh_entsize;
		int strndx = shdr.sh_link;
		for (int i = 0; i < nsymbols; i++)
		{
			GElf_Sym sym;
			if (!gelf_getsym(symtab_data, i, &sym))
			{
				fprintf(stderr, "gelf_getsym() failed for %s: %s\n",
					filename, elf_errmsg(-1));
				goto finish;
			}
			char* name = elf_strptr(e, strndx, sym.st_name);
			if (!name)
			{
				fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
					i, filename, elf_errmsg(-1));
				goto finish;
			}
			if (!strcmp(name, symname))
			{
				result = (char*)sym.st_value;
				goto finish;
			}
		}
	}

finish:
	if (e) elf_end(e);
	if (fd >= 0) close(fd);

	return result;
}

static char* kernel_name = NULL;

WRAP_FUNCTION_BEGIN(calFunctionCreate, int,
	int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5,
	int64_t arg6, int64_t arg7, int64_t arg8, int64_t arg9)

	if (wrapper_funcname)
	{
		kernel_name = (char*)arg1;
		if (!strncmp(wrapper_funcname, kernel_name, strlen(wrapper_funcname)))
		{
			char* regcount_suffix = "_registers_size__";
			char* kernel_name_regcount = (char*)malloc(strlen(kernel_name) + strlen(regcount_suffix) + 1);
			memcpy(kernel_name_regcount, kernel_name, strlen(kernel_name));
			memcpy(kernel_name_regcount + strlen(kernel_name), regcount_suffix, strlen(regcount_suffix) + 1);
			int* regcount = (int*)elfGetSymData(kernel_name_regcount);
			free(kernel_name_regcount);
			if (regcount)
				printf("%s regcount = %d\n", kernel_name, *regcount);
		}
	}

	return __real_calFunctionCreate(
		arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);

WRAP_FUNCTION_END

WRAP_FUNCTION_BEGIN(calFunctionLaunch, int,
	int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5)

	struct timespec start, finish;
	get_time(&start);
	int result = __real_calFunctionLaunch(
		arg0, arg1, arg2, arg3, arg4, arg5);
	get_time(&finish);
	double kernel_time = get_time_diff(&start, &finish);
	if (kernel_name)
		printf("%s kernel time = %f\n", kernel_name, kernel_time);
	else
		printf("kernel time = %f\n", kernel_time);
	return result;

WRAP_FUNCTION_END
