dockerimage ?= das/ssvr
dockerfile ?= Dockerfile.cpu
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

i: install


update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

u: update


run:
	@docker run -it --rm --ipc="host" --network="host" -v $(srcdir)/ssvr:/usr/src/app/ssvr -v $(datadir):/data --entrypoint=/bin/bash $(dockerimage)

r: run


publish:
	@docker image save $(dockerimage) \
	  | pv -N "Publish $(dockerimage) to $(sshopts)" -s $(shell docker image inspect $(dockerimage) --format "{{.Size}}") \
	  | ssh $(sshopts) "docker image load"

p: publish


.PHONY: install i run r update u publish p
