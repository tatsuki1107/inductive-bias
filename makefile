e:
	docker compose exec reseach_container bash

b:
	docker compose build

u:
	docker compose up -d

s:
	docker compose stop

run:
	docker compose run --rm reseach_container sh -c "poetry run python src/main.py"
