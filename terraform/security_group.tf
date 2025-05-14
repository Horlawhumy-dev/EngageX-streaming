# # Security Group for ECS services (Django, Celery, and Flower)
# resource "aws_security_group" "ecs_sg" {
#   name        = "ecs_sg"
#   description = "Security Group for ECS services (Django, Celery, Flower)"
#   vpc_id      = var.vpc_id # Provide your VPC ID

#   # Allow incoming traffic on port 80 (HTTP) and 443 (HTTPS) for web application
#   ingress {
#     from_port   = 80
#     to_port     = 80
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"] # Open to the world (can be restricted based on your needs)
#   }

#   ingress {
#     from_port   = 443
#     to_port     = 443
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"] # Open to the world (can be restricted based on your needs)
#   }

#   # Allow inbound traffic on port 8000 for internal communication between ECS tasks
#   ingress {
#     from_port   = 8000
#     to_port     = 8000
#     protocol    = "tcp"
#     cidr_blocks = ["10.0.0.0/16"] # Private VPC CIDR block (adjust based on your VPC CIDR)
#   }

#   # Allow inbound traffic on the PostgreSQL port (5432) from the ECS tasks (Django app)
#   ingress {
#     from_port   = 5432
#     to_port     = 5432
#     protocol    = "tcp"
#     cidr_blocks = ["10.0.0.0/16"] # Private VPC CIDR block (adjust based on your VPC CIDR)
#   }

#   # Allow inbound traffic on the Redis port (6379) from ECS tasks (Django app, Celery)
#   ingress {
#     from_port   = 6379
#     to_port     = 6379
#     protocol    = "tcp"
#     cidr_blocks = ["10.0.0.0/16"] # Private VPC CIDR block (adjust based on your VPC CIDR)
#   }

#   # Allow inbound traffic for WebSocket (if applicable)
#   ingress {
#     from_port   = 8001
#     to_port     = 8001
#     protocol    = "tcp"
#     cidr_blocks = ["10.0.0.0/16"] # Private VPC CIDR block (adjust based on your VPC CIDR)
#   }

#   # Allow all outbound traffic (you can restrict if needed)
#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1" # All traffic
#     cidr_blocks = ["0.0.0.0/0"]
#   }
# }

# # Security Group for Load Balancer (if you are using one for ECS tasks)
# resource "aws_security_group" "lb_sg" {
#   name        = "lb_sg"
#   description = "Security Group for Load Balancer"
#   vpc_id      = var.vpc_id

#   # Allow incoming traffic on port 80 (HTTP) and 443 (HTTPS) for the load balancer
#   ingress {
#     from_port   = 80
#     to_port     = 80
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"]
#   }

#   ingress {
#     from_port   = 443
#     to_port     = 443
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"]
#   }

#   # Allow inbound traffic on port 80 and 443 from the ECS tasks for the load balancer
#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1" # All traffic
#     cidr_blocks = ["0.0.0.0/0"]
#   }
# }