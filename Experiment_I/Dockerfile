FROM python:3.9-slim

WORKDIR /thesis-zhukova

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /usr/src/app
COPY . .

EXPOSE 11434

#RUN chmod +x /thesis-zhukova/start_services.sh

# Run .sh file
#CMD ["C:/Users/daren/Desktop/MSc Thesis Bau/thesis-zhukova/start_service.sh"]

CMD ["python", "persuasionforgood_pilot_week9-10.py"]
